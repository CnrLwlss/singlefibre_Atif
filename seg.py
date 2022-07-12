import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import time
import os
import feret
import seaborn as sns

makehistograms = False

outdir = "output"
if not os.path.isdir(outdir):
    os.mkdir(outdir)

start = time.time()

# Find bounding box using numpy
# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def class_dict_maker(class_mask, instance_mask):
    unique_colours = np.unique(instance_mask[np.nonzero(instance_mask)])
    classes_dict = {}
    for colour in unique_colours:
        # Get coordinates of a single pixel of that colour
        example_coord = np.transpose((instance_mask==colour).nonzero())[0]
        classes_dict[colour]= class_mask[example_coord[0],example_coord[1]]
    return classes_dict

maskim = Image.open("P03_instance_mask.tiff")
analyseim = Image.open("P03_analysability_mask.tiff")
annotateim = Image.open("P03.tiff")
vdac1 = Image.open("P03_VDAC1.tiff")
dyst = Image.open("P03_Dystrophin.tiff")

maskarr = np.array(maskim,dtype=np.uint16)
analysearr = np.array(analyseim, dtype=np.uint16)
annotatearr = np.array(annotateim,dtype=np.uint8)

# Identify fibres manually classified as non-analysable
classdict = class_dict_maker(analysearr,maskarr)
nonan = [k for k in classdict.keys() if classdict[k]==3]

vdac1arr = np.array(vdac1,dtype=np.uint16)
vdac1arr8b = np.array(np.round(255.0*np.minimum(vdac1arr/np.quantile(vdac1arr,0.99),1.0)),dtype=np.uint8)
vdac1im = Image.fromarray(vdac1arr8b)
#vdac1im.show()
dystarr = np.array(dyst,dtype=np.uint16)
dystarr8b = np.array(np.round(255.0*np.minimum(dystarr/np.quantile(dystarr,0.99),1.0)),dtype=np.uint8)
dystim = Image.fromarray(dystarr8b)
#dystim.show()

values, counts = np.unique(vdac1arr[maskarr>0], axis=0, return_counts=True)
values, counts = np.unique(dystarr[maskarr>0], axis=0, return_counts=True)

unique_colours = np.unique(maskarr)
# Discard black background
unique_colours = [c for c in unique_colours if c!=0]



if makehistograms:
    # Distribution of pixel intensity in preview annotation image
    _, bins, _ = plt.hist(annotatearr[maskarr>0][:,0], bins=np.linspace(0,255,256),color = (1,0,0,0.5), ec=None,label="DYST")
    _ = plt.hist(annotatearr[maskarr>0][:,1], bins=bins, alpha=0.25,color=(0,1,0,0.5),ec=None,label="VDAC1")
    plt.title("All pixels in preview annotation images")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(outdir,'adjustedpixels_ALL.png'), bbox_inches='tight',dpi=300)
    plt.close()

    # Distribution of pixel intensity in raw channel images
    _, bins, _ = plt.hist(dystarr[maskarr>0], bins=np.linspace(0,60,61),color = (1,0,0,0.5), ec=None,label="DYST")
    _ = plt.hist(vdac1arr[maskarr>0], bins=bins, alpha=0.25,color=(0,1,0,0.5),ec=None,label="VDAC1")
    plt.title("All pixels in raw channel images")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(outdir,'pixels_ALL.png'), bbox_inches='tight',dpi=300)
    plt.close()

colnames = ['Colour', 'Class', 'Area', 'Feret_max', 'Feret_min', 'Feret_90_max', 'Feret_90_min','AspectRatio','VDAC1Mean','VDAC1Median','VDAC1Var','DYSTMean','DYSTMedian','DYSTVar']
df = pd.DataFrame(columns=colnames)

for colour in unique_colours:
    # Find pixels belonging to specific fibre
    fibmask = maskarr==colour
    bbox = bbox2(fibmask)

    # Crop arrays to bounding box
    fibarr = np.copy(annotatearr[bbox[0]:bbox[1],bbox[2]:bbox[3],:])
    fibvdac1 = np.copy(vdac1arr[bbox[0]:bbox[1],bbox[2]:bbox[3]])
    fibdyst = np.copy(dystarr[bbox[0]:bbox[1],bbox[2]:bbox[3]])
    
    if(np.sum(fibmask)>2): # Seem to be some tiny annotations?  Why is this?  e.g. colour = 1584
        fibmask2 = fibmask[bbox[0]:bbox[1],bbox[2]:bbox[3]]

        # Delete any non-fibre pixels that remain in bounding box
        fibarr[np.logical_not(fibmask2)] = [0,0,0]

        # Create a suite of statistical summaries for identifying analysable fibres
        pfib = fibarr[fibmask2]
        pvdac1 = fibvdac1[fibmask2]
        pdyst = fibdyst[fibmask2]

        area = np.sum(fibmask2)
        feret_max, feret_min, f90_max, f90_min = feret.all(fibmask2)
        aspectratio = feret_max/max(feret_min,1.0)
        vdac1mean = np.mean(pvdac1)
        vdac1median = np.median(pvdac1)
        vdac1var = np.var(pvdac1)
        dystmean = np.mean(pdyst)
        dystmedian = np.median(pdyst)
        dystvar = np.var(pdyst)

        df = df.append({'Colour':colour, 'Class':classdict[colour], 'Area':area, 'Feret_max':feret_max, 'Feret_min':feret_min,
                        'Feret_90_max':f90_max, 'Feret_90_min':f90_min,'AspectRatio':aspectratio,'VDAC1Mean':vdac1mean,'VDAC1Median':vdac1median,'VDAC1Var':vdac1var,
                        'DYSTMean':dystmean,'DYSTMedian':dystmedian,'DYSTVar':dystvar}, ignore_index=True)

        if makehistograms:
            _, bins, _ = plt.hist(pfib[:,0], bins=np.linspace(0,29,30),color = (1,0,0,0.5), ec=None,label="DYST")
            _ = plt.hist(pfib[:,1], bins=bins, alpha=0.25,color=(0,1,0,0.5),ec=None,label="VDAC1")
            plt.title(f'class_{classdict[colour]:01d}_fibre_{colour:04d}')
            plt.xlabel("Pixel intensity")
            plt.ylabel("Frequency")
            plt.xlim([0,30])
            #plt.ylim([0,300])
            plt.legend(loc="upper right")
            
            
            plt.savefig(os.path.join(outdir,f'class_{classdict[colour]:01d}_fibre_{colour:04d}_hist.jpg'), bbox_inches='tight',dpi=300)
            plt.close()

        # Convert cropped array to image and save to disk
        fibim = Image.fromarray(fibarr, mode="RGB")
        fibim = fibim.resize((fibim.size[0]*5,fibim.size[1]*5),Image.NEAREST)
        fibim.save(os.path.join(outdir,f'class_{classdict[colour]:01d}_fibre_{colour:04d}.jpg'),quality=75)

end = time.time()
total_time = end - start
print("\n"+ str(total_time))

df.to_csv(os.path.join(outdir,'StatisticalSummaries.csv'))

# Can we differentiate between classes using statistical summaries?
# Visualise difference between classes, variable by variable
for var in colnames[2:len(colnames)]:
    sns.stripplot(data=df,x='Class',hue='Class',y=var, size=4, linewidth=0,edgecolor=None,alpha=0.15)
    plt.savefig(os.path.join(outdir,f'stripplot_{var}.jpg'), bbox_inches='tight',dpi=300)
    plt.close()

goodsummaries = ['Area', 'Feret_max', 'Feret_min', 'AspectRatio','VDAC1Mean', 'VDAC1Var', 'DYSTMean', 'DYSTVar']
summ = np.array(df[goodsummaries])
standard = (summ - np.mean(summ,0))/np.std(summ,0)

A = standard[df["Class"]==1,:]
Acolours = df["Colour"][df["Class"]==1]
N = standard[df["Class"]!=1,:]
Ncolours = df["Colour"][df["Class"]!=1]

fig, ax = plt.subplots()
for i,cl in enumerate(Acolours):
    plt.plot(A[i,:], label=cl,color="black",alpha=0.05);
for i,cl in enumerate(Ncolours):
    plt.plot(N[i,:], label=cl,color="red",alpha=0.25);
plt.xticks(range(0,len(goodsummaries)), goodsummaries, rotation='vertical')
fig.tight_layout()
plt.savefig(os.path.join(outdir,'StandardisedSummaries.jpg'), bbox_inches='tight',dpi=300)
plt.close()
    

    


