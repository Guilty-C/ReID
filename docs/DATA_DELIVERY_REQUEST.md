# Data Delivery Request

## Current Status: PARTIAL (Mock Data Only)
- **Available**: 28 mock images for testing
- **Missing**: Full Market-1501 dataset with proper annotations
- **Minimum Viable**: 20+ images per split, 5+ IDs coverage

## Required Dataset Structure
`
dataset_root/
 query/           # Query images
 bounding_box_test/ # Gallery images  
 train/           # Training images (optional for ReID)
 annotations/     # ID, camera, timestamp metadata
`

## Delivery Checklist
- [ ] Dataset archive with SHA256 checksum
- [ ] Image format validation (JPG/PNG)
- [ ] Proper split structure (query/gallery/train)
- [ ] Annotations with ID and camera information
- [ ] Minimum 20 images per split
- [ ] Coverage of at least 5 unique IDs
- [ ] License and compliance documentation

## Next Steps
1. Provide dataset archive or access path
2. Validate structure and annotations
3. Update manifest.csv with real metadata
4. Run full ReID pipeline validation
