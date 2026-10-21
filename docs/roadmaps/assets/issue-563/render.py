"""Render the six English diagrams for RL-Kernel issue #563.

The geometry and visual style follow the approved Chinese roadmap.
Requires reportlab, pymupdf and Pillow; run in the desired output directory.
"""
from pathlib import Path
import math
import fitz
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors

ASSETS = Path(__file__).resolve().parent
INK=colors.HexColor('#1E1E1E'); BLUE=colors.HexColor('#1971C2')
PALE=colors.HexColor('#99C9EF'); GRAY=colors.HexColor('#515151')

class Figure:
    def __init__(self,c,w=1200,h=660):
        self.c=c; self.w=w; self.h=h; self.hatched_regions=[]
    def label(self,x,y,s,size=23,bold=False,color=INK,align='center'):
        c=self.c; c.setFillColor(color);c.setFont('Helvetica' if bold else 'Helvetica',size)
        # Keep thin blue hatching clear of glyphs, without a text-box fill.
        width=pdfmetrics.stringWidth(s,'Helvetica',size)
        left=x-width/2 if align=='center' else (x-width if align=='right' else x)
        baseline=self.h-y-size*.34
        if any(rx <= x <= rx+rw and ry <= y <= ry+rh for rx,ry,rw,rh in self.hatched_regions):
            c.saveState(); c.setFillColor(colors.white); c.setStrokeColor(colors.white)
            c.setLineWidth(1.45)
            text=c.beginText(left,baseline); text.setFont('Helvetica',size)
            text.setTextRenderMode(2); text.textOut(s); c.drawText(text); c.restoreState()
            c.setFillColor(color); c.setFont('Helvetica',size)
        fn={'center':c.drawCentredString,'left':c.drawString,'right':c.drawRightString}[align]
        fn(x,baseline,s)
    def node(self,x,y,w,h,lines,new=False,small=False):
        c=self.c;yy=self.h-y-h
        c.setFillColor(colors.white);c.setStrokeColor(INK);c.setLineWidth(1.8);c.roundRect(x,yy,w,h,13,fill=1,stroke=1)
        if new:
            c.saveState();p=c.beginPath();p.roundRect(x+1,yy+1,w-2,h-2,12);c.clipPath(p,stroke=0,fill=0)
            c.setStrokeColor(PALE);c.setLineWidth(.6)
            for k in range(-int(h),int(w)+int(h),8): c.line(x+k,yy,x+k+h,yy+h)
            c.restoreState();c.setStrokeColor(INK);c.setLineWidth(1.8);c.roundRect(x,yy,w,h,13,stroke=1,fill=0)
            self.hatched_regions.append((x,y,w,h))
            self.label(x-8,y+8,'[+]',size=20,bold=True,color=BLUE,align='right')
        specs=[]
        for idx,l in enumerate(lines):
            if isinstance(l,tuple): specs.append(l)
            else: specs.append((l,24 if idx==0 else 19,idx==0))
        specs=[(s,min(sz,(w-22)/max(pdfmetrics.stringWidth(s,'Helvetica' if (b and b[0]) else 'Helvetica',sz),1)*sz),*(b or [False])) for s,sz,*b in specs]
        total=sum(sz*1.4 for _,sz,*_ in specs)
        cy=y+(h-total)/2
        for line,sz,*b in specs:
            self.label(x+w/2,cy+sz*.68,line,size=sz,bold=b[0] if b else False)
            cy+=sz*1.4
        return (x,y,w,h)
    def group(self,x,y,w,h,title):
        c=self.c;c.setStrokeColor(GRAY);c.setLineWidth(1.4);c.setDash(1.5,5);c.roundRect(x,self.h-y-h,w,h,16,stroke=1,fill=0);c.setDash()
        tw=pdfmetrics.stringWidth(title,'Helvetica',20)
        c.setFillColor(colors.white);c.rect(x+20,self.h-y-13,tw+20,26,fill=1,stroke=0)
        self.label(x+30,y,title,size=20,align='left')
    def arrow(self,pts,blue=False,dash=False,label=None,at=None):
        c=self.c;col=BLUE if blue else INK;c.setStrokeColor(col);c.setFillColor(col);c.setLineWidth(2 if blue else 1.7)
        c.setDash(7,6) if dash else c.setDash()
        p=c.beginPath();p.moveTo(pts[0][0],self.h-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,self.h-y)
        c.drawPath(p);c.setDash()
        a,b=pts[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);r=9
        for off in [-.45,.45]:c.line(b[0],self.h-b[1],b[0]-r*math.cos(ang+off),self.h-(b[1]-r*math.sin(ang+off)))
        if label:
            lx,ly=at;tw=pdfmetrics.stringWidth(label,'Helvetica',18);c.setFillColor(colors.white);c.rect(lx-tw/2-5,self.h-ly-12,tw+10,24,fill=1,stroke=0);self.label(lx,ly,label,size=18,color=col)

def architecture(f):
    f.group(320,142,850,134,'Training and rollout integration')
    f.group(320,301,850,186,'Shared integration and kernel selection')
    f.group(320,521,850,135,'Backends')
    f.node(38,24,280,79,[('Existing: vime integration',23,False),('integrations/orchestrators/vime',16,False)])
    f.node(495,24,500,79,[('New: verl integration',25,False),('integrations/orchestrators/verl',20,False)],True)
    f.arrow([(318,63),(495,63)],blue=True,dash=True,label='Reuse the approach',at=(405,82))
    f.node(355,174,330,81,[('Extend: training integration',23,False),('Init hook + logprob provider',17,False),('engines/train/<backend>',17,False)],True)
    f.node(805,174,330,81,[('Extend: rollout integration',23,False),('vLLM plugin / operator replacement',18,False),('engines/rollout/vllm',17,False)],True)
    f.node(370,330,750,59,[('Extend: shared operator integration',23,False),('integrations/common / engines/*/operators',19,False)],True)
    f.node(370,414,750,59,[('Extend: ops interfaces, registry / plan',23,False),('New: selector / performance implementations',20,False)],True)
    f.node(38,174,225,112,[('New: MoE model specification',22,False),('models/qwen3_moe',18,False),('Model / operator requirements',19,False)],True)
    f.node(38,356,225,110,[('Extend: contracts / platforms',18,False),('Add MoE numerical rules',19,False),('Extend chip capabilities',19,False)],True)
    f.node(360,548,225,64,[('Extend: Ascend backend',22,False),('backends/ascend',18,False)],True)
    f.node(640,548,215,64,[('Existing: CUDA backend',21,False),('backends/cuda',18,False)])
    f.node(910,548,215,64,[('Existing: ROCm backend',21,False),('backends/rocm',18,False)])
    for x,t in [(472,'csrc/ascend'),(747,'csrc/cuda'),(1017,'csrc/rocm')]:f.label(x,638,t,size=19)
    f.arrow([(650,103),(650,174)])
    f.arrow([(840,103),(970,174)])
    f.arrow([(520,255),(520,286),(770,286),(770,330)])
    f.arrow([(970,255),(970,330)])
    f.arrow([(745,389),(745,414)])
    f.arrow([(263,230),(297,230),(297,356),(370,356)],blue=True,dash=True)
    f.arrow([(263,411),(299,411),(299,445),(370,445)],blue=True,dash=True)
    for x in [472,747,1017]:
        f.arrow([(x,473),(x,548)])
        f.arrow([(x,612),(x,624)])
    f.label(150,529,'Reuse the Megatron integration',16)
    f.label(150,559,'Add other engines as needed',16)
    f.label(150,597,'State, EP and tests: see code tables',14)


def dispatcher(f):
    f.group(18,22,1158,164,'Read configuration and registered kernels')
    f.node(48,69,276,86,['Chip and software versions','platforms / SoC / versions'],True)
    f.node(431,69,343,86,['Model, shape and phase','shape / dtype / profile'])
    f.node(880,69,263,86,['Registered kernels','registry / capability'])
    f.node(410,246,380,88,['Find compatible implementations','Check gradients, precision and consistency'])
    f.node(900,250,243,80,['No compatible kernel','Report unsupported configuration'])
    f.node(48,359,276,116,['Kernel measurements','Latency / workspace memory','performance + benchmarks'],True)
    f.node(410,411,380,91,['Select by latency and memory','selector / within the memory limit'],True)
    f.node(898,428,250,126,['Cache the selection','Chip / shape / rules / versions','Reuse on match; reselect on change'],True)
    f.node(410,565,380,76,['Bind kernels for this run','operators / plan / provenance'],True)
    for x,y in [(186,155),(602,155),(1012,155)]:f.arrow([(x,y),(x,208),(602,208),(602,246)])
    f.arrow([(790,290),(900,290)],label='None',at=(846,275))
    f.arrow([(602,334),(602,411)])
    f.arrow([(324,417),(410,451)],blue=True,dash=True)
    f.arrow([(602,502),(602,565)])
    f.arrow([(898,457),(790,457)],blue=True,dash=True,label='Cache hit',at=(845,437))
    f.arrow([(790,603),(1023,603),(1023,554)],blue=True,dash=True,label='Save choice',at=(903,603))
    f.label(186,550,'No measurements: use a valid default',16)
    f.label(186,580,'and mark it as untuned',18)

def speculative(f):
    f.group(20,31,252,558,'Inference engine')
    f.group(309,31,600,585,'RL-Kernel shared interfaces')
    f.node(47,104,199,91,['Draft / MTP','Generate draft tokens'])
    f.node(351,104,517,91,['Target model verifies draft tokens','Attention / MoE / target logprobs'])
    f.node(351,248,258,83,['Find the accepted prefix','Target / draft probabilities / RNG'],True)
    f.node(680,248,189,83,['Resample on rejection','Residual distribution'],True)
    f.node(351,391,517,80,[('Pack retained tokens; apply stop rules',22,False),('On rejection: accepted prefix + resampled token',16,False),('If all accepted: add a bonus token from the target distribution',16,False)],True)
    f.node(351,529,517,65,['Commit accepted state; roll back rejected state','Track version, accepted length and computed-state length'],True)
    f.node(47,413,199,122,['Engine cache','KV / GDN / Conv','Snapshot / commit / rollback'])
    f.node(963,252,212,153,['Rollout output','Retained tokens','Target logprobs','Policy version'])
    f.node(963,508,212,85,['Training recomputes','Same weights / probability rules'])
    f.arrow([(246,150),(351,150)])
    f.arrow([(480,195),(480,248)])
    f.arrow([(609,290),(680,290)],label='Reject',at=(645,311))
    f.arrow([(480,331),(480,391)],label='Accepted prefix',at=(469,366))
    f.arrow([(774,331),(774,391)])
    f.arrow([(610,471),(610,529)])
    f.arrow([(351,561),(284,561),(284,474),(246,474)],blue=True,dash=True)
    f.arrow([(868,560),(939,560),(939,380),(963,380)])
    f.arrow([(1069,405),(1069,508)])
    f.label(147,273,'Draft scheduling / cache blocks',16)
    f.label(147,307,'Connect through the adapter',18)
    f.label(975,637,'Shared by CUDA / ROCm / Ascend',16)

def pd_consistency(f):
    # Lifelines are drawn first so the action boxes cover them cleanly.
    c=f.c;c.setStrokeColor(colors.HexColor('#A5ADB5'));c.setLineWidth(1.2)
    c.line(190,660-86,190,660-574)
    c.line(720,660-86,720,660-631)
    c.line(1057,660-86,1057,660-230)
    c.line(1057,660-444,1057,660-565)
    f.node(50,20,280,64,['Prefill','Pin the weight version'])
    f.node(575,20,290,64,['Decode','Use the same weights'])
    f.node(942,20,230,64,['Training','Use the rollout checkpoint'])
    f.arrow([(190,122),(720,122)],label='Request, versions and KV format',at=(455,104))
    f.node(580,148,280,56,['Allocate receive buffers'],True)
    f.arrow([(720,230),(190,230)],label='Receiver ready',at=(455,212))
    f.node(46,260,288, 62,['Compute the prefix','Attention / MoE'])
    f.arrow([(190,353),(720,353)],blue=True,label='KV / hybrid state + metadata',at=(455,334))
    f.node(565,381,310, 80,['Validate versions and prefix','Confirm required state has arrived'],True)
    f.node(28,497,350, 62,['On failure: report or recompute the prefix'])
    f.arrow([(565,421),(467,421),(467,528),(378,528)],blue=True,dash=True)
    f.node(575,507,290, 62,['Decode / Verify','Continue generation; record logprobs'])
    f.arrow([(720,461),(720,507)],label='Valid',at=(786,483))
    f.group(935,249,245,190,'Training comparison')
    for y,t in [(285,'Same weight version'),(326,'Same tokens / positions / mask'),(367,'Same logprob definition')]:f.label(1057,y,t,16)
    f.node(940,565,237, 77,['Recompute; compare layers','Target logprobs / MoE outputs'],True)
    f.arrow([(720,569),(720,603),(940,603)],blue=True,dash=True,label='tokens / logprob',at=(830,583))
    f.label(265,606,'Generation waits for the full input state',17)

def pd_topology(f):
    f.group(20,25,1160,230,'1P1D: one Prefill node and one Decode node (example)')
    f.node(49,72,448,141,['P: 8 NPUs / EP8','128 experts; 16 per rank','Expert 80 on rank 5'])
    f.node(713,72,431,141,['D: 4 NPUs / EP4','128 experts; 32 per rank','Expert 80 on rank 2'])
    f.arrow([(497,145),(713,145)],blue=True,label='State handoff',at=(605,120))
    f.label(605,177,'KV / hybrid state',size=16)
    f.label(605,202,'+ metadata',size=16)
    f.label(600,236,'Preserve logical expert IDs; validate top-k slots and numerical rules for the same input',19)
    f.group(20,305,1160,290,'Multi-node example: two Prefill nodes and four Decode nodes')
    f.label(257,345,'Prefill: EP16',23,True)
    f.label(942,345,'Decode: EP8',23,True)
    f.node(49,370,399,180,[])
    f.node(732,370,442,180,[])
    f.node(63,384,378, 58,['P-node 0: 8 NPUs','rank 0-7'])
    f.node(63,473,378, 58,['P-node 1: 8 NPUs','rank 8-15'])
    f.node(745,384,194, 58,['D-node 0: 2 NPUs','rank 0-1'])
    f.node(968,384,194, 58,['D-node 1: 2 NPUs','rank 2-3'])
    f.node(745,473,194, 58,['D-node 2: 2 NPUs','rank 4-5'])
    f.node(968,473,194, 58,['D-node 3: 2 NPUs','rank 6-7'])
    f.node(509,394,170,127,[('P to D',24,False),('KV / hybrid state',17,False),('Handoff metadata',17,False),('Target layout',17,False)],True)
    f.arrow([(448,458),(509,458)])
    f.arrow([(679,458),(732,458)])
    f.label(260,564,'Within P: EP dispatch / return',19)
    f.label(950,564,'Within D: EP dispatch / return',19)
    f.label(600,634,'Measure separately: EP within P, P-to-D state transfer, and EP within D',20)

def pd_combine(f):
    f.group(23,25,321,177,'Arrival order (example)')
    for x,t in [(48,'C2'),(120,'C0'),(192,'C3'),(264,'C1')]:f.node(x,81,61,47,[(t,22,True)])
    f.label(181,166,'token / slot / expert in each message',16)
    f.node(411,25,548,177,[],True)
    f.label(685,53,'Write to fixed buffer slots',23,True)
    for j in range(8):
        x=441+(j%4)*125;y=87+(j//4)*57
        f.node(x,y,108,40,[(f'{j} : C{j}',20,False)])
    f.node(1015,42,160,144,['Check this token:','all 8 slots','received'])
    f.arrow([(344,119),(411,119)])
    f.arrow([(959,119),(1015,119)])
    f.group(23,265,1154,385,'Read from the buffer; reduce with a fixed FP32 tree')
    f.arrow([(1095,186),(1095,233),(600,233),(600,300)])
    for x in [153,451,749,1047]:f.arrow([(600,300),(x,300),(x,332)])
    for x,a,b in [(51,0,1),(349,2,3),(647,4,5),(945,6,7)]:f.node(x,332,204,57,[(f'C{a} + C{b}',25,True)])
    f.node(133,454,341, 59,[('(C0 + C1) + (C2 + C3)',22,False)])
    f.node(726,454,341, 59,[('(C4 + C5) + (C6 + C7)',22,False)])
    f.arrow([(153,389),(303,454)]);f.arrow([(451,389),(303,454)])
    f.arrow([(749,389),(896,454)]);f.arrow([(1047,389),(896,454)])
    f.node(444,563,312, 70,['Add both results','Cast at the specified output step'])
    f.arrow([(303,513),(540,563)]);f.arrow([(896,513),(660,563)])
    f.label(205,581,'No pairing by rank',20)
    f.label(205,616,'No arrival-ordered accumulation',18)
    f.label(975,581,'Check duplicate, missing and stale results',16)
    f.label(975,616,'Readiness does not change the sum order',16)

FIGS={'architecture':architecture,'dispatcher':dispatcher,'speculative':speculative,'pd-consistency':pd_consistency,'pd-topology':pd_topology,'pd-combine':pd_combine}

for name,fn in FIGS.items():
    path=ASSETS/f'{name}.pdf';c=canvas.Canvas(str(path),pagesize=(1200,660));fn(Figure(c));c.showPage();c.save()
    d=fitz.open(path);p=d[0];p.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(ASSETS/f'{name}.png')
    im=Image.open(ASSETS/f'{name}.png')
    im.quantize(colors=128,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE).save(ASSETS/f'{name}.png',optimize=True)
    (ASSETS/f'{name}.svg').write_text(p.get_svg_image(),encoding='utf-8');d.close()

