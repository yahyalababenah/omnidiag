const fs=require('fs');
const D=require('docx');
const {Document,Packer,Paragraph,TextRun,ImageRun,Table,TableRow,TableCell,WidthType,ShadingType,HeadingLevel,AlignmentType,LevelFormat,BorderStyle}=D;
const md=fs.readFileSync('FL_REPORT.md','utf8').split('\n');
const FONT='Arial', W=9026;
const run=(t,o={})=>new TextRun({text:t,font:FONT,rightToLeft:true,size:22,...o});
function inline(text,base={}){
  const out=[]; const re=/(\*\*[^*]+\*\*|`[^`]+`)/g; let last=0,m;
  while((m=re.exec(text))){ if(m.index>last) out.push(run(text.slice(last,m.index),base));
    const tok=m[0];
    if(tok.startsWith('**')) out.push(run(tok.slice(2,-2),{...base,bold:true}));
    else out.push(new TextRun({text:tok.slice(1,-1),font:'Courier New',size:20,...base}));
    last=m.index+tok.length;}
  if(last<text.length) out.push(run(text.slice(last),base));
  return out;
}
const P=(children,o={})=>new Paragraph({bidirectional:true,alignment:AlignmentType.START,spacing:{after:120,line:300},children,...o});
function pngSize(f){const b=fs.readFileSync(f);return [b.readUInt32BE(16),b.readUInt32BE(20)];}
const kids=[]; const numbering=[{reference:'bul',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.START,style:{paragraph:{indent:{left:540,hanging:270}}}}]}];
let olCount=0, curOl=null;
function newOl(){olCount++;curOl='ol'+olCount;numbering.push({reference:curOl,levels:[{level:0,format:LevelFormat.DECIMAL,text:'%1.',alignment:AlignmentType.START,style:{paragraph:{indent:{left:540,hanging:270}}}}]});}
let i=0, inOl=false;
while(i<md.length){
  const l=md[i];
  if(l.startsWith('```')){ const code=[]; i++; while(i<md.length&&!md[i].startsWith('```')){code.push(md[i]);i++;} i++;
    for(const c of code) kids.push(new Paragraph({alignment:AlignmentType.LEFT,spacing:{after:0},shading:{type:ShadingType.CLEAR,fill:'F2F2F2'},children:[new TextRun({text:c||' ',font:'Courier New',size:18})]}));
    kids.push(new Paragraph({children:[]})); inOl=false; continue;}
  if(l.startsWith('|')){ const rows=[]; while(i<md.length&&md[i].startsWith('|')){ if(!/^\|[\s\-|:]+\|$/.test(md[i])) rows.push(md[i].split('|').slice(1,-1).map(s=>s.trim())); i++; }
    const n=rows[0].length, cw=Math.floor(W/n), widths=Array(n).fill(cw); widths[n-1]=W-cw*(n-1);
    const b={style:BorderStyle.SINGLE,size:4,color:'BBBBBB'}, borders={top:b,bottom:b,left:b,right:b};
    kids.push(new Table({width:{size:W,type:WidthType.DXA},columnWidths:widths,visuallyRightToLeft:true,
      rows:rows.map((r,ri)=>new TableRow({tableHeader:ri===0,children:r.map((c,ci)=>new TableCell({borders,width:{size:widths[ci],type:WidthType.DXA},margins:{top:60,bottom:60,left:90,right:90},
        shading:ri===0?{type:ShadingType.CLEAR,fill:'D9EAD3'}:undefined,
        children:[new Paragraph({bidirectional:true,alignment:AlignmentType.CENTER,children:inline(c,{size:19,bold:ri===0})})]}))}))}));
    kids.push(new Paragraph({spacing:{after:120},children:[]})); inOl=false; continue;}
  let m;
  if((m=l.match(/^!\[[^\]]*\]\(([^)]+)\)/))){ const f=m[1]; const [w,h]=pngSize(f); const pw=560, ph=Math.round(pw*h/w);
    kids.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:160},keepNext:false,children:[new ImageRun({type:'png',data:fs.readFileSync(f),transformation:{width:pw,height:ph},altText:{title:f,description:f,name:f}})]})); i++; inOl=false; continue;}
  if(l.startsWith('# ')){kids.push(new Paragraph({heading:HeadingLevel.TITLE,bidirectional:true,alignment:AlignmentType.CENTER,spacing:{after:200},children:[run(l.slice(2),{bold:true,size:40})]}));i++;continue;}
  if(l.startsWith('## ')){kids.push(new Paragraph({heading:HeadingLevel.HEADING_1,bidirectional:true,spacing:{before:320,after:140},keepNext:true,children:[run(l.slice(3).replace(/`/g,""),{bold:true,size:30,color:'1F4E79'})]}));i++;inOl=false;continue;}
  if(l.startsWith('### ')){kids.push(new Paragraph({heading:HeadingLevel.HEADING_2,bidirectional:true,spacing:{before:240,after:120},keepNext:true,children:[run(l.slice(4),{bold:true,size:26,color:'2E75B6'})]}));i++;inOl=false;continue;}
  if(/^\s*- /.test(l)){const lvl=/^\s{2,}- /.test(l); kids.push(P(inline(l.replace(/^\s*- /,'')),{numbering:{reference:'bul',level:0},indent:lvl?{left:1080,hanging:270}:undefined})); i++; inOl=false; continue;}
  if((m=l.match(/^(\d+)\. (.*)/))){ if(!inOl){newOl();inOl=true;} kids.push(P(inline(m[2]),{numbering:{reference:curOl,level:0}})); i++; continue;}
  if(l.trim()===''){i++;continue;}
  inOl=false; kids.push(P(inline(l.replace(/^> ?/,'')))); i++;
}
const doc=new Document({numbering:{config:numbering},styles:{default:{document:{run:{font:FONT,size:22}}}},
  sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1134,bottom:1134,left:1440,right:1440}}},children:kids}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync('FL_REPORT.docx',b);console.log('ok',b.length)});
