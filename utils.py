
# RGB to gray scale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# Uniform filtering function

## first uniform filter function: too slow (square complexity)

def uniform_filter(tab,size=(8,8)):
    [sx,sy]=tab.shape
    unif=np.zeros([sx,sy])
    
    dx_down=size[0]//2
    dy_down=size[1]//2
    dx_up=(size[0]//2-np.ceil(size[0]/2))*int(size[0]%2==0)+np.ceil(size[0]/2)
    dy_up=(size[1]//2-np.ceil(size[1]/2))*int(size[1]%2==0)+np.ceil(size[1]/2)

    #cm=np.cumsum(np.cumsum(tab,axis=1),axis=0)
    
    for i in range(sx):
        for j in range(sy):
            ax=np.maximum(0,i-dx_down)
            bx=np.minimum(sx-1,i+dx_up)
            ay=np.maximum(0,j-dy_down)
            by=np.minimum(sy-1,j+dy_up)
            unif[i,j]=np.mean(tab[ax:bx,ay:by])
            #unif[i,j]=cm[bx,by]-cm[bx,ay]-cm[by,ax]+tab[ax,ay]
    return unif

## second uniform filter function: still too slow (square complexity)

def unif_filter_2(im,s=(8,8)):
    cm=np.cumsum(np.cumsum(im,axis=1),axis=0)

    dx_down=int(s[0]//2)
    dy_down=int(s[1]//2)
    dx_up=int((s[0]//2-np.ceil(s[0]/2))*int(s[0]%2==0)+np.ceil(s[0]/2))
    dy_up=int((s[1]//2-np.ceil(s[1]/2))*int(s[1]%2==0)+np.ceil(s[1]/2))

    unif=np.zeros((32,32))

    a=np.concatenate((cm[:,:s[1]],cm[:,:-s[1]]),axis=1)
    b=np.concatenate((cm[:s[0],],cm[:-s[0],:]),axis=0)
    c=np.concatenate((cm[:s[0],:-s[0]],cm[:-s[0],:-s[0]]),axis=0)
    c=np.concatenate((c[:,:s[0]],c),axis=1)

    for x in range(dx_down,32-dy_up):
        for y in range(dy_down,32-dy_up):
            #x_i=x-dx_down
            x_j=x+dx_up
            #y_i=y-dy_down
            y_j=y+dy_up
            unif[x,y]=((cm-a-b+c)/(s[0]*s[1]))[x_j-1,y_j-1]

    return unif

## third uniform filter function: matrix trick to make it faster

def unif_filter(im,s=(8,8)):
    [sx,sy]=im.shape
    cm=np.cumsum(np.cumsum(im,axis=1),axis=0)

    dx_down=int(s[0]//2)
    dy_down=int(s[1]//2)
    dx_up=int((s[0]//2-np.ceil(s[0]/2))*int(s[0]%2==0)+np.ceil(s[0]/2))
    dy_up=int((s[1]//2-np.ceil(s[1]/2))*int(s[1]%2==0)+np.ceil(s[1]/2))

    a=np.concatenate((np.fliplr(cm[:,:s[1]]),cm[:,:-s[1]]),axis=1)
    b=np.concatenate((np.flipud(cm[:s[0],]),cm[:-s[0],:]),axis=0)
    c=np.concatenate((cm[:s[0],:-s[1]],cm[:-s[0],:-s[1]]),axis=0)
    c=np.concatenate((c[:,:s[1]],c),axis=1)

    cu=(cm-a-b+c)/np.cumsum(np.cumsum(np.ones((sx,sy)),axis=0),axis=1)
    cu[dx_down:sx-dx_up+1,dy_down:sy-dy_up+1]=((cm-a-b+c)/(s[0]*s[1]))[s[0]-1:sx,s[1]-1:sy]

    return cu




