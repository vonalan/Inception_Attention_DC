import main 

video_dir = 'videos'
split_dir = 'splits'

results = main.create_video_lists(video_dir, split_dir, sround=1)
print(results)