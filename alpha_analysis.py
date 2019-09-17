from postemq_analysis import get_contingency_matrix_one_label


def contingency_matrix_mle_alpha(posterior_probs, pair, alphas, minecore, cr=True):
    alpha_r, alpha_p = alphas[pair]
    if cr:
        us_post = minecore.user_posteriors(alpha_r)[pair[0]]
        return get_contingency_matrix_one_label(minecore.y_arr[pair[0]], posterior_probs[pair[0]]), \
               get_contingency_matrix_one_label(minecore.y_arr[pair[0]], us_post)
    else:
        us_post = minecore.user_posteriors(alpha_p)[pair[1]]
        return get_contingency_matrix_one_label(minecore.y_arr[pair[1]], posterior_probs[pair[1]]), \
               get_contingency_matrix_one_label(minecore.y_arr[pair[1]], us_post)


def normalize_contingency_table(cont_table, n_test_set_docs):
    return dict(map(lambda kv: (kv[0], kv[1] / n_test_set_docs), cont_table.items()))
