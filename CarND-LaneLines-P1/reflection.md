# **Finding Lane Lines on the Road** 

## Pipeline

My pipeline concludes 6 step:

1. Turn RGB image into gray image.
2. Add gauss blur to cut noise in the gray image.
3. Use canny edge detector to find edges.
 - calculate images gradient
 - use non-maximum suppression to thin image
 - use double threshold to filter edges
4. Fix the region of interest using polygon
5. Use hough transform to find lines
6. Extract line from hough transformation


In order to draw a single line on the left and right lanes, I modified the draw_lines() to draw_lines2(). 

```python
def draw_lines2(img, lines, color=[255, 0, 0], thickness=2):
    # split lines into left and right parts
    left = []
    right = []
    for x1,y1,x2,y2 in lines[:,0,:]:
        k = (y1-y2)/(x1-x2)
        b = y1 - k*x1
        if k<0:
            left.append([k,b])
        elif k>0:
            right.append([k,b])
    left = np.array(left)
    right = np.array(right)
    
    # get left and right mean lines
    try:
        left_k, _ = np.median(left,axis=0)
        right_k, _ = np.median(right,axis=0)
        _, left_b = np.mean(left,axis=0)
        _, right_b = np.mean(right,axis=0)
    except:
        return
    if not left_k or not left_b:
        return
        
    ylim = img.shape[0]-1
    ydist = 200
    ## func 
    f_left = lambda y: int((y-left_b)/left_k)
    f_right = lambda y: int((y-right_b)/right_k)
    ## points
    y_lim = img.shape[0]
    try:
        left_points = [(f_left(ylim),ylim),(f_left(ylim-ydist),ylim-ydist)]
        right_points = [(f_right(ylim),ylim),(f_right(ylim-ydist),ylim-ydist)]
    except:
        return 
    ## plot
    cv2.line(img, left_points[0], left_points[1], color, thickness)
    cv2.line(img, right_points[0], right_points[1], color, thickness)  
```

## Shortcomings

1. The added lines' locations aren't so precise compared with original locations in the image.
2. The added lines' locations aren't stable. It flashes sometimes.
3. The added lines disappear in the challenge video. 

## Possible improvements

1. Filter lines extracted form hough transform to make it more precise compared with original land lanes.
2. Convolution with adjacent frame with gauss-like kernel to make land lines stable.
3. Adjust parameters of ROI (decrease region size in the corners) and hough transform to  find precise lines.

# Thanks

Thank you for your review. I am a little frustrated that my work isn't as good as examples. Could you give me some advices to do works better as the example video.

