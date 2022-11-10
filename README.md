# College-Pointcloud-Compression
A framework made in Python for Point Cloud compression using Computer Vision techniques.

# Command to create videos from plot images
ffmpeg -i img%d.jpg -c:v libx264 -r 2 -pix_fmt yuv420p out.mp4