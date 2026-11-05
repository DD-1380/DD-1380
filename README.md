# DD-1380 Medical Form Transcription Pipeline

An OCR-powered document transcription pipeline for converting DD Form 1380 (Tactical Combat Casualty Care Card) images into structured JSON data.

This project automates the extraction of handwritten and printed medical information from DD-1380 casualty cards using modern computer vision and Optical Character Recognition (OCR) techniques. The extracted information is transformed into a standardized Badok JSON format for downstream processing.

---

## Overview

The pipeline performs the following steps:

1. Upload a photograph of a DD-1380 casualty card.
2. Align and correct the document using DocScan.
3. Run OCR using DocTR to recognize text.
4. Extract predefined fields from the aligned document.
5. Convert extracted fields into the Badok JSON schema.
6. Display the resulting structured data.

The project is designed to support future integration with machine learning models for checkbox detection and automated scoring.

---

## Pipeline

```text
Raw Image
    │
    ▼
DocScan
(Document Alignment)
    │
    ▼
DocTR OCR
(Text Recognition)
    │
    ▼
Field Extraction
(Crop + OCR)
    │
    ▼
Field Mapping
(Badok Converter)
    │
    ▼
Badok JSON
```

---

## Features

- Automatic document alignment
- OCR using DocTR
- Region-based field extraction
- Conversion to structured Badok JSON
- Gradio web interface for testing
- Modular pipeline architecture
- Easily extendable field mapping

---

## Project Structure

```
project/
│
├── app.py                 # Gradio interface
├── pipeline.py            # Main processing pipeline
├── extract_fields.py      # Field extraction logic
├── converter.py           # Converts extracted fields to Badok JSON
├── schema.json            # Badok schema definition
├── field_map.json         # Maps OCR fields to Badok fields
├── image_transform.py
├── overlay.py
└── README.md
```

---

## Technologies Used

- Python
- OpenCV
- DocTR
- DocScan
- NumPy
- Gradio
- JSON

---

## How It Works

### 1. Document Alignment

The uploaded image is processed using DocScan to remove perspective distortion and align the form with the reference template.

### 2. Optical Character Recognition

DocTR performs OCR on the aligned image to recognize printed and handwritten text.

### 3. Field Extraction

The template (`source.json`) contains the location of every field on the DD-1380 form.

Each field is cropped individually and passed through OCR to obtain its text value.

Example:

```python
{
    "field_name": "John Doe",
    "field_service": "Army",
    "field_unit": "1-75 Rangers"
}
```

### 4. Field Conversion

The extracted field names are mapped into the Badok schema.

Example:

```python
field_name
        ↓
NAME

field_service
        ↓
SERVICE
```

Result:

```json
{
    "id": null,
    "values": {
        "NAME": "John Doe",
        "SERVICE": "Army",
        "UNIT": "1-75 Rangers"
    },
    "total_score": 0,
    "score_components": {}
}
```

---

## Future Improvements

- Checkbox detection using computer vision
- Confidence scoring for OCR predictions
- Automatic JSON export
- REST API integration
- Support for additional military medical forms

---

## My Contributions

- Implemented the field-to-schema conversion module (`converter.py`)
- Designed the mapping between OCR field names and the Badok JSON schema
- Integrated the conversion stage into the document processing pipeline
- Developed the schema initialization and default value handling
- Improved the modular architecture for future expansion
- Assisted in debugging OCR field extraction and schema mapping

---

## License

This project was developed as part of the DD-1380 Medical Form Transcription project.

License information will be added in the future.
