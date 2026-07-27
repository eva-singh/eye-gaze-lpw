import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# train_ssd.py
import os,csv,torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as T
from models.ssd import get_model

class EyeGazeDataset(Dataset):
    def __init__(self,img_dir,label_dir):
        self.samples=[]
        labels={}
        for f in os.listdir(label_dir):
            if f.endswith(".txt"):
                labels[os.path.splitext(f)[0]]=os.path.join(label_dir,f)
        self.transforms=T.Compose([T.Resize((300,300)),T.ToTensor()])
        for root,_,files in os.walk(img_dir):
            for f in files:
                if not f.lower().endswith(".jpg"): continue
                stem=os.path.splitext(f)[0]
                lp=labels.get(stem) or labels.get(stem.replace("frame_",""))
                if lp: self.samples.append((os.path.join(root,f),lp))
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        ip,lp=self.samples[idx]
        img=self.transforms(Image.open(ip).convert("RGB"))
        cls,xc,yc,bw,bh=map(float,open(lp).readline().split())
        s=300
        xc*=s; yc*=s; bw*=s; bh*=s
        x1=max(0,xc-bw/2); y1=max(0,yc-bh/2)
        x2=min(s,xc+bw/2); y2=min(s,yc+bh/2)
        tgt={"boxes":torch.tensor([[x1,y1,x2,y2]],dtype=torch.float32),
             "labels":torch.tensor([1],dtype=torch.int64)}
        return img,tgt

def collate_fn(b): return tuple(zip(*b))

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=EyeGazeDataset("dataset/images","dataset/labels")
    dl=DataLoader(ds,batch_size=4,shuffle=True,collate_fn=collate_fn)
    model=get_model().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    out="runs/ssd"; os.makedirs(out,exist_ok=True)
    losses=[]; epochs=5
    for e in range(epochs):
        model.train(); total=0
        for imgs,tgts in dl:
            imgs=[i.to(device) for i in imgs]
            tgts=[{k:v.to(device) for k,v in t.items()} for t in tgts]
            ld=model(imgs,tgts); loss=sum(ld.values())
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item()
        avg=total/len(dl); losses.append(avg)
        print(f"Epoch {e+1}/{epochs} Loss {avg:.4f}")
        torch.save(model.state_dict(),os.path.join(out,f"ssd_epoch_{e+1}.pth"))
    torch.save(model.state_dict(),os.path.join(out,"ssd_final.pth"))
    with open(os.path.join(out,"loss_history.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Epoch","Loss"])
        [w.writerow([i+1,l]) for i,l in enumerate(losses)]
    plt.figure(figsize=(8,5))
    plt.plot(range(1,epochs+1),losses,marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
    plt.title("SSD Training Loss")
    plt.savefig(os.path.join(out,"loss_curve.png"))
    plt.close()
    print("Done.")

if __name__=="__main__":
    main()
