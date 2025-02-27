# Dimension Reduction

## Distance

|         | `train`-`valid`           |  `train`-`test`           |
|------------------|--------------------|--------------------|
| **base**        | ![alt text](base_imgfiles/image.png) | ![alt text](base_imgfiles/image-1.png) |
| **pca** `dim = 50`        | ![alt text](base_imgfiles/image-2.png) | ![alt text](base_imgfiles/image-3.png) |

## plot
|         |         | `train`-`valid`           |  `train`-`test`           |pca | 
|------------------|------------------|--------------------|--------------------|--------------------|
| **t-SNE**|    scatter       | ![alt text](base_imgfiles/image-4.png) | ![alt text](base_imgfiles/image-5.png) | x |
| **t-SNE**|    scatter       | ![alt text](base_imgfiles/image-12.png) | ![alt text](base_imgfiles/image-13.png) | o |
| **t-SNE** | density  | ![alt text](base_imgfiles/image-6.png) | ![alt text](base_imgfiles/image-7.png) | x |
| **t-SNE** | density  | ![alt text](base_imgfiles/image-14.png) | ![alt text](base_imgfiles/image-15.png) | o |
| **UMAP**  |    scatter     | ![alt text](base_imgfiles/image-8.png) | ![alt text](base_imgfiles/image-9.png) | x |
| **UMAP**  |    scatter     | ![alt text](base_imgfiles/image-16.png) | ![alt text](base_imgfiles/image-17.png) | o |
| **UMAP** | density  | ![alt text](base_imgfiles/image-10.png) | ![alt text](base_imgfiles/image-11.png) | x |
| **UMAP** | density  | ![alt text](base_imgfiles/image-18.png) | ![alt text](base_imgfiles/image-19.png) | o |


## 🥼 pca several times
p = Cumulative explained variance
|         |         | `train`-`valid`           |  `train`-`test`           |pca | 
|------------------|------------------|--------------------|--------------------|--------------------|
| **UMAP**|    scatter   2D    | ![alt text](base_imgfiles/image-20.png) | ![alt text](base_imgfiles/image-24.png) | 50 |
| **UMAP**|    scatter    2D     | ![alt text](base_imgfiles/image-21.png) | ![alt text](base_imgfiles/image-25.png) | 30 |
| **UMAP**|    scatter     2D    | ![alt text](base_imgfiles/image-22.png) | ![alt text](base_imgfiles/image-26.png) | 10 |
| **UMAP**|    scatter      2D   | ![alt text](base_imgfiles/image-23.png) | ![alt text](base_imgfiles/image-27.png) | 5 |
