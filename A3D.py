import torch
from torch.utils import data

class A3DDataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.all_features = os.path.join(self.cfg.DATASET.ROOT, 'features')
        self.all_labels = os.path.join(self.cfg.DATASET.ROOT, 'labels')
        self.num_abnormal = 810 # Total number of abnormal videos in Training Dataset.
        self.num_normal = 800 # Total number of Normal videos in Training Dataset.

        self.batch_size = self.cfg.DATASET.BATCH # 60
        self.n_exp=self.batch_size/2  # Number of abnormal and normal videos

    def __getitem__(self):
        # We assume the features of abnormal videos and normal videos are located in two different folders.
        Abnor_list_iter = np.random.permutation(num_abnormal)
        Abnor_list_iter = Abnor_list_iter[num_abnormal-n_exp:] # Indexes for randomly selected Abnormal Videos
        Norm_list_iter = np.random.permutation(num_normal)
        Norm_list_iter = Norm_list_iter[num_normal-n_exp:]     # Indexes for randomly selected Normal Videos


        AllVideos_Path = AbnormalPath
        def listdir_nohidden(AllVideos_Path):  # To ignore hidden files
            file_dir_extension = os.path.join(AllVideos_Path, '*_C.txt')
            for f in glob.glob(file_dir_extension):
                if not f.startswith('.'):
                    yield os.path.basename(f)

        All_Videos=sorted(listdir_nohidden(AllVideos_Path))
        All_Videos.sort()
        AllFeatures = []  # To store C3D features of a batch
        print("Loading Abnormal videos Features...")

        Video_count=-1
        for iv in Abnor_list_iter:
            Video_count=Video_count+1
            VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
            # f = np.load(VideoPath)
            # num_feat = len(words) / 4096
            # # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Note that
            # # we have already computed C3D features for the whole video and divide the video features into 32 segments. Please see Save_C3DFeatures_32Segments.m as well

            # count = -1
            # VideoFeatues = []
            # for feat in xrange(0, num_feat):
            #     feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            #     count = count + 1
            #     VideoFeatues.append(feat_row1)
            # VideoFeatues = torch.stack(VideoFeatues)
            VideoFeatues = np.load(VideoPath)
            AllFeatures.append(VideoFeatues)
            
        print("Loading Normal videos...")
        AllVideos_Path =  NormalPath

        def listdir_nohidden(AllVideos_Path):  # To ignore hidden files
            file_dir_extension = os.path.join(AllVideos_Path, '*_C.txt')
            for f in glob.glob(file_dir_extension):
                if not f.startswith('.'):
                    yield os.path.basename(f)

        All_Videos = sorted(listdir_nohidden(AllVideos_Path))
        All_Videos.sort()

        for iv in Norm_list_iter:
            VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
            # f = open(VideoPath, "r")
            # words = f.read().split()
            # feat_row1 = np.array([])
            # num_feat = len(words) /4096   # Number of features to be loaded. In our case num_feat=32, as we divide the video into 32 segments.

            # VideoFeatues = []
            # for feat in xrange(0, num_feat):
            #     feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])                
            #     VideoFeatues.append(feat_row1)
            # VideoFeatues = torch.stack(VideoFeatues)
            VideoFeatues = np.load(VideoPath)
            AllFeatures.append(VideoFeatues)
        AllFeatures = torch.stack(AllFeatures)
        print("Features  loaded")

        AllLabels = np.zeros(32*self.batch_size, dtype='uint8')
        th_loop1=n_exp*32
        th_loop2=n_exp*32-1

        for iv in xrange(0, 32*self.batch_size):
                if iv< th_loop1:
                    AllLabels[iv] = int(0)  # All instances of abnormal videos are labeled 0.  This will be used in custom_objective to keep track of normal and abnormal videos indexes.
                if iv > th_loop2:
                    AllLabels[iv] = int(1)   # All instances of Normal videos are labeled 1. This will be used in custom_objective to keep track of normal and abnormal videos indexes.

        return AllFeatures, AllLabels
