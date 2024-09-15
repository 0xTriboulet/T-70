# T-70: Facial Verification Malware POC

## Project Overview

**T-70** is a Rust project inspired by the **Cyberdyne Series 70 Automated Fighting Infantry Unit** ("The Terminator"). The project performs runtime facial verification using **OpenCV** through the **opencv-rust bindings** and integrates a **CascadeClassifier** with **MobileNetV2** models for face detection and similarity comparison. The system is designed to run as a standalone executable, making it ideal for deployment in environments where real-time facial verification is essential.

### Key Features:
- **VM Detection**: The project includes virtual machine detection logic to determine whether the executable is running in a virtualized environment. If a VM is detected, the executable self-deletes.
- **Face Detection and Comparison**: The system captures images using a connected camera, detects faces, and compares embeddings from the live feed with a reference image to compute similarity scores.
- **Automated Action**: Based on the similarity score, the system can run shellcode if a match is found.

## Project Structure

- **`main.rs`**: The main entry point of the project.
- **Modules**:
    - `camera`: Handles camera initialization and input.
    - `classifier`: Manages the initialization of the face detection classifier.
    - `face_detection`: Responsible for detecting faces and processing the reference image.
    - `similarity`: Calculates similarity scores between the detected faces and reference faces.

- **External Functions**:
    - Functions like `GetProcessCountViaSnapShot`, `GetUniqueUserCountViaSnapshot`, and `VmDetection` are integrated into the project for advanced runtime features.

## Dependencies

This project depends on the following:

- **OpenCV**: Specifically, a statically compiled version of the OpenCV library is required.
- **opencv-rust bindings**: Provides the Rust bindings to OpenCV. Check the [opencv-rust GitHub repository](https://github.com/twistedfall/opencv-rust) for more information.

## Setup Instructions

### Step 1: Compile OpenCV
Before building the project, ensure that OpenCV is compiled statically. Follow the OpenCV documentation for setting up a statically compiled version of the library on your system.

### Step 2: Download Models
In the `scripts` directory, you will find Python scripts that help download offline versions of the MobileNetV2 and CascadeClassifier models. These are required to run the facial detection and comparison components.

### Step 3: Build the Project
Once OpenCV is compiled and the models are downloaded, you can build the project using Cargo:

```bash
cargo build --release
```

### Step 4: Run the Executable
After building the project, you can run the executable to start the face detection and verification process:

```bash
./target/release/t70
```

## Special Thanks

Special thanks to [twistedfall](https://github.com/twistedfall) for maintaining the `opencv-rust` project, which was critical for the development of the T-70 system.

## License

T-70 is licensed under the MIT License.

---

*Inspired by the Cyberdyne Series 70 Automated Fighting Infantry Unit ("The Terminator").*  
More information: [Cyberdyne Series 70 Terminator](https://terminator.fandom.com/wiki/T-70)

