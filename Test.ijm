// 1. Create GUI dialog
Dialog.create("PyTorch Inference Plugin");
Dialog.addFile("Python Interpreter", "D:/Software/Anaconda/Scripts/conda.exe");
Dialog.addDirectory("Input Images", "E:/Paper_Writing/04UniSR/ImageJ/plugins/UniSR_Plugin/dataset/SIM_MTs/");
Dialog.addDirectory("Output Directory", "E:/Paper_Writing/04UniSR/ImageJ/plugins/UniSR_Plugin/Output/");
Dialog.addFile("Model File (.pth)", "E:/Paper_Writing/04UniSR/ImageJ/plugins/UniSR_Plugin/checkpoints/MTs/netG_epoch_500.pth");
Dialog.show();

// 2. Get user input
conda_exe = Dialog.getString();
input_path = Dialog.getString();
output_path = Dialog.getString();
model_path = Dialog.getString();

// 3. Set the path for the Python processing script (it is recommended to keep it relative to the Macro)
python_script = getDirectory("plugins") + "UniSR_Plugin\\scripts\\test.py";

// 4. Replace backslashes "\" with forward slashes "/" to prevent escape issues
input_path = replace(input_path, "\\", "/");
model_path = replace(model_path, "\\", "/");
output_path = replace(output_path, "\\", "/");
python_script = replace(python_script, "\\", "/");

input_path = "\"" + input_path + "\"";
model_path = "\"" + model_path + "\"";
output_path = "\"" + output_path + "\"";
python_script = "\"" + python_script + "\"";

// 5. Build the execution command
command = conda_exe + " run -n torch1.13 python " + python_script + " " + input_path + " " + output_path + " " + model_path;

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
