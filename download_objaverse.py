import objaverse
objaverse.__version__
import multiprocessing
import random 

def create_filtered_list():
    uids = []
    #uidss = objaverse.load_uids()
    uid_lvis = objaverse.load_lvis_annotations()
    annotations = objaverse.load_annotations()
    #print(annotations)
    for uid,annotation in annotations.items():
        for lvis_annotation,lvis_uid  in uid_lvis.items():
            #print(annotation['animationCount'])
            if(annotation['animationCount']==0 and annotation['categories'] is not None):
                if uid in lvis_uid:
                    uids.append(uid)
        #pass
    print(len(uids))        
    return uids        


if __name__ == "__main__":
    processes = multiprocessing.cpu_count()
    random.seed(42)
    # for l in list:
    #     id_with_extension = l.split('/')[1]
    #     id_without_extension = id_with_extension.split('.')[0]
    #     uids.append(id_without_extension)

    #uids = objaverse.load_uids()
    #random_object_uids = random.sample(uids, 50)
    #print("Total uids loaded: ",len(uids))
    uids = create_filtered_list()
    #print(uids)
    objects = objaverse.load_objects(
        uids=uids,
        download_processes=processes
    )
    #print(objects)





