// 1. Create GUI dialog
Dialog.create("PyTorch Training Plugin");
Dialog.addFile("Conda Interpreter", "D:/Software/Anaconda/Scripts/conda.exe");
Dialog.addDirectory("Training Dataset", "E:/Paper_Writing/04UniSR/ImageJ/plugins/UniSR_Plugin/dataset/SIM_MTs_single");
Dialog.addDirectory("Model Saving Path", "E:/Paper_Writing/04UniSR/ImageJ/plugins/UniSR_Plugin/checkpoints/MTs");
Dialog.addDirectory("Visualization during Training", "E:/Paper_Writing/04UniSR/ImageJ/plugins/UniSR_Plugin/results/MTs");
Dialog.addFile("Pre-trained Model File (.pth)", "E:/Paper_Writing/04UniSR/ImageJ/plugins/UniSR_Plugin/checkpoints/Pretrained_model/Simu_Line_SR_ep479_Unet.pth");
Dialog.addString("Image Size", "256");
Dialog.addString("Batch Size", "1");
Dialog.addString("Epoch Count", "0");
Dialog.addString("Total Epochs", "10");
Dialog.addString("Generator Learning Rate", "8e-5");
Dialog.addString("Learning Rate Policy(step, plateau, cosine)", "cosine");
Dialog.addString("LR Decay Iterations", "50");
Dialog.show();

// 2. Get user input
conda_exe = Dialog.getString();
training_path = Dialog.getString();
model_path = Dialog.getString();
graph_path = Dialog.getString();
pretrain_path = Dialog.getString();
img_size = Dialog.getString();
batch_size = Dialog.getString();
epoch_count = Dialog.getString();
nEpochs = Dialog.getString();
generatorLR = Dialog.getString();
lr_policy = Dialog.getString();
lr_decay_iters = Dialog.getString();

// 3. Set the path for the Python processing script
python_script = getDirectory("plugins") + "UniSR_Plugin\\scripts\\train.py";

// 4. Replace backslashes "\" with forward slashes "/" to prevent escape issues
training_path = replace(training_path, "\\", "/");
model_path = replace(model_path, "\\", "/");
graph_path = replace(graph_path, "\\", "/");
pretrain_path = replace(pretrain_path, "\\", "/");
python_script = replace(python_script, "\\", "/");

training_path = "\"" + training_path + "\"";
model_path = "\"" + model_path + "\"";
graph_path = "\"" + graph_path + "\"";
pretrain_path = "\"" + pretrain_path + "\"";
python_script = "\"" + python_script + "\"";

// 5. Construct execution command
//command = conda_exe + " run -n torch python " + python_script +  " " + training_path + " " +  model_path + " " + graph_path + " " + pretrain_path + " " + img_size + batch_size + epoch_count + nEpochs + generatorLR + lr_policy + lr_decay_iters;
command = conda_exe + " run -n torch1.13 python " + python_script + " " +
          training_path + " " + model_path + " " + graph_path + " " + pretrain_path + " " +
          img_size + " " + batch_size + " " + epoch_count + " " +
          nEpochs + " " + generatorLR + " " + lr_policy + " " + lr_decay_iters;
          
// 6. Determine the operating system type; Windows requires cmd execution, while Linux/macOS can run directly
sys_info = getInfo("os.name");
if (indexOf(sys_info, "Windows") != -1) {
    command = "cmd /c " + command;  // For Windows, use cmd to execute
}

// 7. Execute the Python inference command
print("Running command: " + command);
output = exec(command);
print("Command output: " + output);

// 8. Wait for the process to complete
wait(5000);  // Add a delay to ensure Python execution is complete
