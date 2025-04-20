# Doc Scan

Doc Scan is a Python-based project for image processing and computer vision tasks. It leverages OpenCV for image manipulation and FastAPI for providing a web-based API to process images.

It's not a production suited project but still... :D

## Features

- **Pre-Processing**:
  - Resizing, denoising, edge detection, HSV filtering, GrabCut segmentation, and morphological operations.
  - Thresholding using Otsu's method.

- **Processing**:
  - Corner detection using Hough Line Transform and Contour-based methods.
  - Perspective transformation for bird's-eye view extraction.

- **Post-Processing**:
  - Image enhancement, filtering, sharpening, and cleaning.

- **FastAPI Integration**:
  - Provides an API endpoint to process uploaded images and return results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AmirKafi/Doc-Scan.git
   cd OpenCV-Prac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Python 3.10 or higher installed.

## Usage

### Running the FastAPI Server

1. Start the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. Access the API at `http://localhost:8000`.

3. Use the `/process-image` endpoint to upload an image and process it.

### Example API Request

- **Endpoint**: `/process-image`
- **Method**: POST
- **Request**: Upload an image file.
- **Response**: Processed image as a downloadable file.

### Running Individual Scripts

- Use the modules in the `pre_processors`, `processors`, and `PostProcessor` directories for specific image processing tasks.

## Project Structure

```
OpenCV-Prac/
├── app.py                     # FastAPI application
├── requirements.txt           # Python dependencies
├── vercel.json                # Vercel deployment configuration
├── PostProcessor/             # Post-processing modules
├── pre_processors/            # Pre-processing modules
├── processors/                # Processing modules
├── static/                    # Static files (e.g., favicon)
└── README.md                  # Project documentation
```

## Deployment

This project can be deployed on [Vercel](https://vercel.com/) using the provided `vercel.json` configuration.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for the computer vision library.
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [scikit-learn](https://scikit-learn.org/) for clustering algorithms.

## Contact

For any questions or feedback, feel free to reach out to the project maintainer.