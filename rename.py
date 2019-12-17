import os

for idx, filename in enumerate(os.listdir("road_pic")):
    if filename == ".DS_Store":
        continue

    dst = "lane_" + str(idx) + ".jpg"
    src = 'road_pic/' + filename
    dst = 'road_pic/' + dst

    # rename() function will
    # # rename all the files
    os.rename(src, dst)

    print(dst)
