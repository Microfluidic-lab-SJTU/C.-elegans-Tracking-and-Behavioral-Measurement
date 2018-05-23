__author__='Edmond'
import configparser
import cv2 
import numpy as np
import SimpleITK as sitk
from lib.myshow import myshow, myshow3d
from matplotlib import pyplot as plt
from  PIL import Image,ImageFilter

def process_config(conf_file):
	"""
		read configuration from files and saved to a dict
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'Global':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'ActiveContour':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params
	
def frame_diff(gray_img,*args):
	diff_frame1=cv2.absdiff(gray_img,args[0])
	diff_frame0=cv2.absdiff(args[1],gray_img)
	return cv2.bitwise_and(diff_frame1,diff_frame0)
	
def concatenate(img1,gray_img):
	gray_img_=np.stack([gray_img,gray_img,gray_img],axis=2)
	concat=np.concatenate([img1,gray_img_],axis=1)
	return concat
	
def thresh_otsu(img,*args):
	#th=cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,4)
	ret,th=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return th
	
def resize_and_gray(img,return_color=False):
	w,h,c=img.shape
	shape=(int(h/2),int(w/2))
	img=cv2.resize(img,shape)
	gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if return_color:
		return img,gray_img
	else:
		return gray_img

def open_op(img,it=1):
	kernel = np.ones((3,3), np.uint8)
	for i in range(it):
		img = cv2.dilate(cv2.erode(img, kernel, iterations=1), kernel, iterations=1)
	return img

def plt_show(img,title=None):
	plt.figure(figsize=(10,10)) 
	if title!=None:
		plt.title(title)
	if len(img.shape)==2:
		cmap=plt.get_cmap('gray')
		plt.imshow(img,cmap=cmap)
	else:
		plt.imshow(img)
	plt.axis('off')
def fill_holes(img):
    '''
    fill holes and filter small component with small area
    '''
    #print np.unique(img)   
    h,w=img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img.copy(),mask,(0,0),255,0,0,8)
    mask[mask==1]=255
    #print np.unique(mask)   
    mask=255-mask[1:h+1,1:w+1]
    img_filled=cv2.bitwise_or(mask,img)
    return img_filled
	
def shape_filter(img,thresh=500):
    contour_list=[]
    (_,cnts, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        Area = cv2.contourArea(c)
        if Area>thresh:
            contour_list.append(c)
    mask=np.zeros(img.shape)
    mask=cv2.drawContours(mask,contour_list,-1,255,-1) 
    return mask
def get_init_contour(img):
	'''
		deprecated function
		Don't use it
	'''
	reverse=255-img
	dist_transform = cv2.distanceTransform(reverse,cv2.DIST_L2,3)
	_,ret=cv2.threshold(dist_transform,5,255,cv2.THRESH_BINARY)
	return 255-ret
	
class geodesicActiveContourSegementation:

	def __init__(self,params):
		self.params=params
		self.PropagationScaling=params['propagationscaling']
		self.CurvatureScaling=params['curvaturescaling']
		self.AdvectionScaling=params['advectionscaling']
		self.MaximumRMSError=params['maximumrmserror']
		self.NumberOfIterations=params['numberofiterations']
		self.Dist_threshold=params['dist_threshold']
		self.Sigma=params['sigma']
		self.geodesicActiveContour = sitk.GeodesicActiveContourLevelSetImageFilter()
		self.geodesicActiveContour.SetPropagationScaling(self.PropagationScaling)
		self.geodesicActiveContour.SetCurvatureScaling(self.CurvatureScaling)
		self.geodesicActiveContour.SetAdvectionScaling(self.AdvectionScaling)
		self.geodesicActiveContour.SetMaximumRMSError(self.MaximumRMSError)
		self.geodesicActiveContour.SetNumberOfIterations(self.NumberOfIterations)
		self.gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
		self.gradientMagnitude.SetSigma(self.Sigma)
		
	def get_init_contour(self,img): 
		reverse=255-img
		dist_transform = cv2.distanceTransform(reverse,cv2.DIST_L2,3)
		_,ret=cv2.threshold(dist_transform,self.Dist_threshold,255,cv2.THRESH_BINARY)
		ret=255-ret
		shape_filtered=shape_filter(ret.astype(np.uint8))
		shape_filtered=shape_filtered.astype(np.uint8)
		holes=fill_holes(shape_filtered)
		init_contour=cv2.morphologyEx(holes,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations = 7)
		return init_contour
		
	def get_init_contour_v2(self,img):
		'''
			params:
				img: img is a the diff of original img and backgroud
		'''
		edge=cv2.Canny(img,7,60)
		reverse=255-edge
		dist_transform = cv2.distanceTransform(reverse,cv2.DIST_L2,3)
		_,ret=cv2.threshold(dist_transform,4,255,cv2.THRESH_BINARY_INV)
		shape_filtered=shape_filter(ret.astype(np.uint8),thresh=2000)
		shape_filtered=shape_filtered.astype(np.uint8)
		reverse=255-shape_filtered
		dist_transform = cv2.distanceTransform(reverse,cv2.DIST_L2,3)
		_,init_contour=cv2.threshold(dist_transform,4,255,cv2.THRESH_BINARY_INV)
		init_contour=init_contour.astype(np.uint8)
		return init_contour
	def execute_v1(self,img,foreground):
		init_contour=self.get_init_contour(foreground)
		init_contour = sitk.GetImageFromArray(init_contour)
		init_contour = sitk.SignedMaurerDistanceMap(init_contour, 
						insideIsPositive=False, useImageSpacing=False)
		featureImage=sitk.GetImageFromArray(img)
		featureImage = sitk.BoundedReciprocal(self.gradientMagnitude.Execute(featureImage))
		featureImage = sitk.Cast( featureImage, init_contour.GetPixelID()) 
		levelset = self.geodesicActiveContour.Execute( init_contour, featureImage )
		contour=levelset<0
		return contour
	def execute(self,diff):
		featureImage=sitk.GetImageFromArray(diff)
		featureImage = sitk.BoundedReciprocal(self.gradientMagnitude.Execute(featureImage))
		bk=sitk.GetArrayFromImage(featureImage)
		bk=((bk/bk.max())*255).astype(np.uint8)
		otsu=thresh_otsu(bk)
		dist_transform = cv2.distanceTransform(otsu,cv2.DIST_L2,3)
		_,retu=cv2.threshold(dist_transform,5,255,cv2.THRESH_BINARY_INV)
		retu=retu.astype(np.uint8)
		(_,cnts, hier) = cv2.findContours(retu.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		areas=[(int(cv2.contourArea(c)),h[3]==-1) for (c,h) in zip(cnts,hier[0])]
		areas=np.array(areas)
		outer_contour_idx=np.where((areas[:,0]>self.params['min_contour_area'])&(areas[:,1]>0))
		inter_contour_idx=np.where((areas[:,0]>self.params['min_hole_area'])&(areas[:,1]<1))
		outer_contour=[cnts[i] for i in outer_contour_idx[0]]
		inter_contour=[cnts[i] for i in inter_contour_idx[0]]
		mask=np.zeros(diff.shape)
		mask=cv2.drawContours(mask,outer_contour,-1,255,-1) 
		mask=cv2.drawContours(mask,inter_contour,-1,0,-1)
		init_contour_ = sitk.GetImageFromArray(mask.astype(np.uint8))
		init_contour_1 = sitk.SignedMaurerDistanceMap(init_contour_, insideIsPositive=False, useImageSpacing=False)
		featureImage = sitk.Cast( featureImage, init_contour_1.GetPixelID())
		levelset = self.geodesicActiveContour.Execute(init_contour_1, featureImage )
		contour=levelset<0
		return contour,otsu
class PIL_filter:
	def __init__(self,filter_type):
		filter_type_dict={'EDGE_ENHANCE':ImageFilter.EDGE_ENHANCE,
							'EDGE_ENHANCE_MORE':ImageFilter.EDGE_ENHANCE_MORE,
							'FIND_EDGES':ImageFilter.FIND_EDGES}
		self.filter_type=filter_type_dict[filter_type]
	def filter(self,img):
		shape=img.shape
		pil_im = Image.fromarray(img)  
		pil_im=pil_im.filter(self.filter_type)
		pil_im=pil_im.convert("L")
		data=pil_im.getdata()
		data=np.matrix(data)
		img=np.reshape(data,shape)
		return img
	def filter_type(self,filter_type):
		self.filter_type=filter_type_dict[filter_type]

class show_img:
	def __init__(self):
		self.imgs=[]
	def append(self,img,name):
		self.imgs.append([img,name])
	def show(self):
		assert(self.imgs is not None)
		for img,name in self.imgs:
			cv2.imshow(name,img)
		k=cv2.waitKey(0)
		if k==27:
			cv2.destroyAllWindows()
class frame_factory:
	def __init__(self,params):
		self.params=params
		self.cap=cv2.VideoCapture(params['video_path'])
		self.map={}
		self.num_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		_,self.map[0]=self.cap.read()
		self.tail=0
		self.head=0
		self.size=(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		self.fps = self.cap.get(cv2.CAP_PROP_FPS)
	def __getitem__(self,index):
		if index>=self.tail and index<=self.head:
			return self.map[index]
		elif index>self.head:
			num_add=(index-self.head)
			assert num_add<20 and index<self.num_frames
			for i in range(1,num_add+1):
				_,self.map[self.head+i]=self.cap.read()	
			self.head=index
			if len(self.map)>20:
				num_del=len(self.map)-20
				for i in range(num_del):
					self.map.pop(self.tail+i)
				self.tail+=num_del
			return self.map[index]
		else:
			raise Exception("error")
	
	def reset(self):
		self.cap=cv2.VideoCapture(self.params['video_path'])
		self.map={}
		_,self.map[0]=self.cap.read()
		self.tail=0
		self.head=0
	def set_file_name(self,name):
		self.writer=cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'XVID'), self.fps, self.size)
	def write(self,frame):
		self.writer.write(frame)
	def write_done(self):
		self.writer.release()
class watershed_segement:
	def __init__(self):
		pass
	