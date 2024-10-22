# Multi-Layer Perceptron to Identify Handwritten Numbers  

### Project Overview  
- **Language**: C++  
- **Libraries**: Standard Library only  
- **Purpose**: This project demonstrates the principles of neural networks from scratch, focusing on simplicity and accessibility for beginners.  

### Future Plans  
This is the third neural network model I’ve built, with more to come. My goal is to contribute accessible, beginner-friendly source code, as I've noticed a lack of resources covering neural networks in this way.  

### Contributions  
I welcome all suggestions, improvements, or feedback! Feel free to reach out or create an issue if you have ideas for enhancements.  

---

### Dependencies  
- **Required**: `stb_image.h`  
This project uses the `stb_image` header file from Sean Barrett’s repository to extract pixel values from PNG images. You can find it here: [stb_image on GitHub](https://github.com/nothings/stb).  

---

### Steps to Compile  
1. **Install** the `letter_rec.cpp` file.  
2. **Download** the `stb_image.h` file from Sean Barrett’s GitHub repo.  
3. **Place** the `stb_image.h` file in the same directory or a subdirectory (if it’s in a subdirectory, modify the `#include` path accordingly).  
4. **Setup** the handwritten number dataset using the following structure:  
   ```  
   ...path_to_dataset...\hand_numbers\dataset\x\x\  
   ```  
   Where `x` is the label (e.g., 0–9) corresponding to each number to be trained. The program will automatically iterate through the files and extract pixel values.  

---

Thank you for checking out my project!
