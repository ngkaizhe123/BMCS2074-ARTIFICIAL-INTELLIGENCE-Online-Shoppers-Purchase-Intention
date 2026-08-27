# Draw.io Import Guide

## How to Use the .drawio File

### Option 1: Direct Import (Recommended)
1. Open [draw.io](https://app.diagrams.net/) in your browser
2. Click "Open Existing Diagram"
3. Select "Device" or navigate to your file location
4. Choose the file: `shopper_intention_pipeline_flowchart.drawio`
5. The flowchart will load automatically with all elements properly formatted

### Option 2: File Import
1. Open draw.io (web or desktop version)
2. Go to **File → Open From → Device**
3. Navigate to your project folder
4. Select `shopper_intention_pipeline_flowchart.drawio`
5. Click "Open"

### Option 3: Copy-Paste Method
If the direct import doesn't work:
1. Open the `.drawio` file in a text editor
2. Copy all the XML content
3. In draw.io, go to **Edit → Edit Diagram**
4. Paste the XML content
5. Click "Apply"

## Features of the Draw.io File

### Flowchart Elements:
- **START/END nodes**: Green/Red ovals
- **Process nodes**: Blue rounded rectangles
- **Decision node**: Orange diamond (for outlier removal decision)
- **Model nodes**: Dark blue rectangles
- **SMOTE node**: Pink rectangle

### Layout:
- Traditional top-to-bottom flow
- Proper branching for outlier removal decision
- Three parallel model training paths
- Converging paths to evaluation and end

### Customization:
You can easily customize in draw.io:
- Change colors by selecting elements and using the format panel
- Adjust text content by double-clicking on elements
- Modify connections by dragging the endpoints
- Add additional elements from the shape library

## Export Options from Draw.io

Once you have the flowchart open in draw.io, you can export it as:

### Common Formats:
- **PNG**: Best for documents and presentations
- **SVG**: Best for web and scalability
- **PDF**: Best for printing and sharing
- **JPEG**: Compressed image format

### Export Steps:
1. Go to **File → Export As**
2. Choose your preferred format
3. Adjust resolution/scale if needed
4. Click "Export" or "Download"

## Troubleshooting

### If the file doesn't open:
- Make sure you're using a recent version of draw.io
- Try opening it in a text editor first to verify the XML is intact
- Use the copy-paste method as a fallback

### If elements are misaligned:
- Use the "Arrange" menu in draw.io to align elements
- Use the grid snapping feature for precise positioning
- Select all elements (Ctrl+A) and use "Format → Align" options

### If text is hard to read:
- Increase font size in the format panel
- Change font colors for better contrast
- Adjust element sizes to accommodate text

## Additional Tips

### Best Practices:
- Use the "Auto-layout" feature for automatic arrangement
- Save frequently (Ctrl+S) to avoid losing changes
- Use layers to organize complex diagrams
- Add comments or annotations for clarity

### Collaboration:
- Share the draw.io link for real-time collaboration
- Export to PNG/SVG for including in reports
- Use version control for the .drawio file

## File Structure

The `.drawio` file contains:
- **XML structure**: Compatible with draw.io format
- **Styled elements**: Pre-formatted with colors and fonts
- **Connected paths**: Properly linked flowchart arrows
- **Text formatting**: HTML-formatted labels with subtext

This should open perfectly in draw.io and give you a fully editable traditional flowchart of your ML pipeline!