import google.generativeai as genai
import PIL.Image
import json
import os
import time

def extract_medical_data_with_gemini(image_path):
    API_KEY = "AIzaSyCRSk8YXz-Pxn7xD8j9p3JvTs665hij-Hg"
    genai.configure(api_key=API_KEY)
    
    # Load the image
    try:
        img = PIL.Image.open(image_path)
        print(f"✅ Image loaded: {image_path}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Use the latest available models that support vision
    models_to_try = [
        'models/gemini-2.0-flash',  # Latest flash model
        'models/gemini-2.0-flash-001',
        'models/gemini-2.5-flash',
        'models/gemini-2.5-flash-preview-05-20',
        'models/gemini-flash-latest',  # Always points to latest
        'models/gemini-pro-latest',   # Always points to latest pro
    ]
    
    print(f"🔄 Processing: {os.path.basename(image_path)}")
    
    prompt = """
    Extract ALL medical data from this nerve conduction study report and return as VALID JSON only.

    Return EXACTLY this structure:
    {
      "Motor_Nerve_Conduction_Studies": [
        {
          "Nerve_Muscles": "string",
          "Stimulus_Site": "string", 
          "Latency_ms": "string",
          "Distance_cm": "string",
          "Amplitude_mv": "string",
          "NCV_ms": "string"
        }
      ],
      "Sensory_Nerve_Conduction_Studies": [
        {
          "Nerve": "string",
          "Recording_Site": "string",
          "Stimulation_Site": "string",
          "Latency_ms": "string", 
          "Distance_cm": "string",
          "Amplitude_uv": "string",
          "NCV_ms": "string"
        }
      ],
      "Electromyography": [
        {
          "Muscles": "string",
          "Fibs": "string",
          "Psw": "string",
          "Others": "string",
          "Amp": "string",
          "Duration": "string",
          "Polys": "string", 
          "Recruit": "string",
          "Interference": "string"
        }
      ]
    }

    Extract EVERY row from all tables. Be precise with numbers and labels. Return ONLY JSON, no other text.
    """
    
    for model_name in models_to_try:
        try:
            print(f"🔄 Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content([prompt, img])
            
            print("✅ Response received!")
            
            # Clean the response to get pure JSON
            response_text = response.text.strip()
            print("Raw response preview:", response_text[:200] + "..." if len(response_text) > 200 else response_text)
            
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1].strip()
            else:
                json_text = response_text
                
            # Parse JSON
            data = json.loads(json_text)
            print(f"✅ Success with {model_name}")
            return data
            
        except Exception as e:
            print(f"❌ Failed with {model_name}: {e}")
            continue
    
    return None

def get_all_images_from_current_folder():
    """Get all image files from current folder"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    current_folder = os.getcwd()
    
    image_files = []
    for file in os.listdir(current_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    return image_files

def main():
    print("🎯 Medical Report Extraction Started")
    print("=" * 50)
    
    # Get all images from current folder
    image_files = get_all_images_from_current_folder()
    
    if not image_files:
        print("❌ No image files found in current folder")
        print("💡 Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
        return
    
    print(f"📁 Found {len(image_files)} images in folder:")
    for img in image_files:
        print(f"   • {img}")
    
    all_results = {}
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"📊 Processing image {i}/{len(image_files)}: {image_file}")
        print(f"{'='*60}")
        
        # Add delay between requests to avoid rate limits
        if i > 1:
            print("⏳ Waiting 15 seconds to avoid rate limits...")
            time.sleep(15)
        
        result = extract_medical_data_with_gemini(image_file)
        
        if result:
            # Save to combined results
            all_results[image_file] = result
            
            # Save individual JSON file
            json_filename = f"result_{os.path.splitext(image_file)[0]}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Individual result saved to: {json_filename}")
            
            # Print summary
            print(f"\n📊 EXTRACTION SUMMARY for {image_file}:")
            print(f"   • Motor Nerves: {len(result.get('Motor_Nerve_Conduction_Studies', []))}")
            print(f"   • Sensory Nerves: {len(result.get('Sensory_Nerve_Conduction_Studies', []))}")
            print(f"   • EMG Entries: {len(result.get('Electromyography', []))}")
        else:
            print(f"❌ Failed to extract data from: {image_file}")
            all_results[image_file] = {"error": "Extraction failed"}
    
    # Save combined results
    with open("all_extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 ALL DONE! Processed {len(image_files)} images")
    print("💾 Combined results saved to: 'all_extracted_data.json'")
    
    # Final summary
    successful = sum(1 for result in all_results.values() if "error" not in result)
    print(f"📈 Successfully extracted: {successful}/{len(image_files)} images")

if __name__ == "__main__":
    main()