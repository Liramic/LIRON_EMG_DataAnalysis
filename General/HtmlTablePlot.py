from dataclasses import dataclass
from enum import Enum
import base64

class ColType(Enum):
    Text = 0
    Photo = 1

def toHtmlColumn(s):
    return f"<td align='center'>{s}</td>"

def toHtmlPhotoColumn(b64Photo):
    return toHtmlColumn(f'<img src="data:image/png;base64, {b64Photo}" style="display: block; max-width:400px; max-height:400px; width: auto; height: auto;" />')

@dataclass
class HtmlCol:
    Type : ColType
    Value : str

    def toColumn(self) -> str :
        if ( self.Type == ColType.Text ):
            return toHtmlColumn(self.Value)
        elif ( self.Type == ColType.Photo ):
            return toHtmlPhotoColumn(self.Value)


def initHtmlDictionary(titles):
    d = dict()
    d["titles"] = ["row_id"] + titles
    return d

def addRow(d : dict, row : list):
    if ( not "rows" in d ):
        d["rows"] = list()
    new_id = len(d["rows"])
    d["rows"] += [[HtmlCol(ColType.Text, str(new_id))] + row]


def PlotDictionary(dict):
    html = "<html><table letter-spacing='1px'  style='table-layout: fixed; width:100%;' border='1px solid black' >"
    #Add titles:
    html += "<tr>"
    for title in dict["titles"]:
        html += f"<th>{title}</th>"
    html += "</tr>"

    #Add rows
    for row in dict["rows"]:
        html+="<tr>"
        for item in row:
            html+= item.toColumn()
        html+="</tr>"
    
    html += "</table></html>"
    return html

def toImageHtmlCol(bytes):
    return HtmlCol(ColType.Photo, base64.b64encode(bytes).decode('ascii'))

def toImageHtmlColFromPath(p):
    with open(p, "rb") as f:
        bytes = f.read()
    return toImageHtmlCol(bytes)

def toTextHtmlCol(obj):
    return HtmlCol(ColType.Text, str(obj))

def SaveHtmlFile(dic, filename):
    html = PlotDictionary(dic)
    with open(filename, "w") as f:
        f.write(html)

if __name__=="__main__":
    #usage example:
    d = initHtmlDictionary(["i","i*2", "image"])
    photos = [r"C:\Liron\DataEmg\Done\20122022_0830\A.jpeg", r"C:\Liron\Code\LIRON_EMG_DataAnalysis\1.png"]

    for i in range(2):
        addRow(d, [HtmlCol(ColType.Text, str(i)), HtmlCol(ColType.Text, str(i*2)), toImageHtmlColFromPath(photos[i]) ])
    SaveHtmlFile(d, "2.html")
