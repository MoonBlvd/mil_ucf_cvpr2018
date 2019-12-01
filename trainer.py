def do_train(model, data_loader):

    AllClassPath='/newdata/UCF_Anomaly_Dataset/Dataset/CVPR_Data/C3D_Features_Txt/Train/'
    # AllClassPath contains C3D features (.txt file)  of each video. Each text file contains 32 features, each of 4096 dimension
    output_dir='/newdata/UCF_Anomaly_Dataset/Dataset/CVPR_Data/Trained_Models/TrainedModel_MIL_C3D/'
    # Output_dir is the directory where you want to save trained weights
    weights_path = output_dir + 'weights.mat'
    # weights.mat are the model weights that you will get after (or during) that training
    model_path = output_dir + 'model.json'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    All_class_files= listdir(AllClassPath)
    All_class_files.sort()
    loss_graph =[]
    num_iters = 20000
    total_iterations = 0
    batchsize=60
    time_before = datetime.now()

    for it_num in range(num_iters):

        AbnormalPath = os.path.join(AllClassPath, All_class_files[0])  # Path of abnormal already computed C3D features
        NormalPath = os.path.join(AllClassPath, All_class_files[1])    # Path of Normal already computed C3D features
        inputs, targets=load_dataset_Train_batch(AbnormalPath, NormalPath)  # Load normal and abnormal video C3D features
        batch_loss =model.train_on_batch(inputs, targets)
        loss_graph = np.hstack((loss_graph, batch_loss))
        total_iterations += 1
        if total_iterations % 20 == 1:
            print("These iteration=" + str(total_iterations) + ") took: " + str(datetime.now() - time_before) + ", with loss of " + str(batch_loss))
            iteration_path = output_dir + 'Iterations_graph_' + str(total_iterations) + '.mat'
            savemat(iteration_path, dict(loss_graph=loss_graph))
        if total_iterations % 1000 == 0:  # Save the model at every 1000th iterations.
            weights_path = output_dir + 'weightsAnomalyL1L2_' + str(total_iterations) + '.mat'
            save_model(model, model_path, weights_path)

