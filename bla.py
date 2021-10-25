with open('data/pos_processed_gnps_shuffled.failed_sdf') as fl,\
     open('data/pos_processed_gnps_shuffled_with_3d.sdf', 'w') as outfl,\
     open('data/pos_processed_gnps_shuffled.txt') as infl:
    failed_entries = [int(l) for l in fl.readlines()]
    for i, l in enumerate(infl.readlines()):
        if i not in failed_entries:
            outfl.write(l)
