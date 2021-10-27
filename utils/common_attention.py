
def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   dropout_rate=0.0,
                                   image_shapes=None,
                                   save_weights_to=None,
                                   name=None,
                                   make_image_summary=True,
                                   cache=False,
                                   allow_memory=False,
                                   hard_attention_k=0,
                                   gumbel_noise_weight=0.0):
  """Calculate relative position-aware dot-product self-attention.

  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.

  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_relative_position: an integer specifying the maximum distance between
        inputs that unique position embeddings should be learned for.
    dropout_rate: a floating point number.
    image_shapes: optional tuple of integer scalars.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    name: an optional string.
    make_image_summary: Whether to make an attention image summary.
    cache: whether use cache mode
    allow_memory: whether to assume that recurrent memory is in use. If True,
      the length dimension of k/v/bias may be longer than the queries, and it is
      assumed that the extra memory entries precede the non-memory entries.
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.

  Returns:
    A Tensor.

  Raises:
    ValueError: if max_relative_position is not > 0.
  """
  if not max_relative_position:
    raise ValueError("Max relative position (%s) should be > 0 when using "
                     "relative self attention." % (max_relative_position))
  with tf.variable_scope(
      name, default_name="dot_product_attention_relative",
      values=[q, k, v]) as scope:

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape, unless memory is enabled.
    if not cache and not allow_memory:
      q.get_shape().assert_is_compatible_with(k.get_shape())
      q.get_shape().assert_is_compatible_with(v.get_shape())

    # Use separate embeddings suitable for keys and values.
    depth = k.get_shape().as_list()[3]
    length_k = common_layers.shape_list(k)[2]
    length_q = common_layers.shape_list(q)[2] if allow_memory else length_k
    relations_keys = _generate_relative_positions_embeddings(
        length_q, length_k, depth, max_relative_position,
        "relative_positions_keys", cache=cache)
    relations_values = _generate_relative_positions_embeddings(
        length_q, length_k, depth, max_relative_position,
        "relative_positions_values", cache=cache)

    # Compute self attention considering the relative position embeddings.
    logits = _relative_attention_inner(q, k, relations_keys, True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if hard_attention_k > 0:
      weights = harden_attention_weights(weights, hard_attention_k,
                                         gumbel_noise_weight)
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if not tf.get_variable_scope().reuse and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return _relative_attention_inner(weights, v, relations_values, False)
