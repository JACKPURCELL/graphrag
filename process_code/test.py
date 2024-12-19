def calculate_need_to_keep(num_indirect_adv_texts, num_text_per_root, num_keep_indirect):
    need_to_keep = []
    for i in range(0, num_indirect_adv_texts, num_text_per_root):
        need_to_keep.extend(range(i, i + num_keep_indirect))
    return need_to_keep    


if __name__ == '__main__':
    num_indirect_adv_texts = 10
    num_text_per_root = 10
    num_keep_indirect = 5
    need_to_keep = calculate_need_to_keep(num_indirect_adv_texts, num_text_per_root, num_keep_indirect)
    print(need_to_keep)
    print(len(need_to_keep))