def write_neg_file():
    with open('neg.txt', 'w') as f:
        for i in range(1, 551):
            f.write('negative/{}.jpg\n'.format(i))
write_neg_file()
