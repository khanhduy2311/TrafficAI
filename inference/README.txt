======================================================================
     			CV - Fisheye Detection 
======================================================================

SYSTEM OVERVIEW
----------------------------------------------------------------------
This solution implements a high-performance, multi-threaded pipeline
for object detection. It is heavily optimized with custom C++ extensions
for pre-processing and result fusion, which are compiled automatically
during the Docker image build.

The workflow for evaluation is a simple, three-step process:
1. Build the Docker image.
2. Run the container, mounting the provided test data.
3. Execute the one-time engine build script, then the inference script.

----------------------------------------------------------------------
STEP 1: Build the Docker Image
----------------------------------------------------------------------
This step installs all dependencies and compiles the custom C++
extensions. This is the only build step required.

From the root directory containing the Dockerfile, run:

  docker build -t infer .

----------------------------------------------------------------------
STEP 2: Run the Container for Evaluation
----------------------------------------------------------------------
The evaluators will run the container, mounting a host directory that
contains the test data (images and ground truth JSON) to the `/data`
directory inside the container.

**Official Evaluation `docker run` Command:**
  docker run -it --rm --runtime=nvidia \
    -v /path/to/evaluation_data:/data \
    -v /path/to/output_dir:/output \
    infer

After running the command, you will be inside the container's shell
at the `/app` working directory.

----------------------------------------------------------------------
STEP 3: Execute Scripts (Inside the Container)
----------------------------------------------------------------------

**3.1 - Run Inference**
   Run the main inference script. It will
   read images from `/data/images` and write the results to
   `/output/submission.json`.

   **Official Inference Command:**
     python3 final_inference.py \
       -i /data/images \
       -o /output/submission.json \
       -d cuda

   **Note:**  
   - If testing on `fisheye1keval`, make sure to use `ensemble_configs_1keval.json`.  
   - Otherwise, use the default `ensemble_configs.json`.  
   - You can pass it via `--params-config ensemble_configs_1keval.json`

**3.2 - (Optional) Run Evaluation with Ground Truth**
   To test the full scoring flow, use the `--ground-truths-path` argument,
   pointing to the ground truth file within the mounted `/data` directory.

   **Example Evaluation Command:**
     python3 final_inference.py \
       -i /data/images \
       -o /output/submission.json \
       -d cuda \
       --ground-truths-path /data/gt.json \
       --max-fps 25.0
