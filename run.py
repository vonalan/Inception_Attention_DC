import main 

# video_dir = 'videos'
# split_dir = 'splits'

video_dir = r'E:\Users\kingdom\HMDB51\hmdb51_org'
split_dir = r'E:\Users\kingdom\HMDB51\testTrainMulti_7030_splits'

results = main.create_video_lists(video_dir, split_dir, sround=1)
# print(results)

for key in results:
    val = results[key]
    print("%s %s %s %s %2d %2d %3d %s %3d"%(key, (16-len(key)) * '-', val['dir'], (16-len(val['dir'])) * '-',
                                     len(val['training']), len(val['testing']), len(val['validation']), 16*'-',
                                            len(val['training'])+len(val['testing'])))