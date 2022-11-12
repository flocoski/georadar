#!/usr/bin/env python

from importlib.resources import path
import tkinter.font as tkFont
from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter import Label, ttk, filedialog
from tkinter.filedialog import askopenfile
from traceback import clear_frames
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import struct
from PIL import Image, ImageDraw
import tkinter.font as font
import pandas as pd 
import pickle


font_family = "Helvetica"
font_size = {"small": 9, "normal": 10, "large": 11, "Large": 12}
text_color = "#000000"
background_color = "#FFFFFF"

def entry(parent):
    font_style = tkFont.Font(family = font_family, size = font_size["normal"], weight = "normal")
    return Entry(parent, bg = background_color, fg = text_color, font = font_style)

def label(parent, label):
    font_style = tkFont.Font(family = font_family, size = font_size["normal"], weight = "normal")
    return Label(parent, text=label ,bg = background_color, fg = text_color, font = font_style)

def label_frame(parent, label):
    font_style = tkFont.Font(family = font_family, size = font_size["normal"], weight = "normal")
    labelframe=LabelFrame(parent, text=label, bg=background_color, fg = text_color, font = font_style, relief = "raised")
    return labelframe

def frame(parent):
    frame = Frame(parent, bg = background_color, relief = "raised")
    return frame

def combobox(parent, options):
    font_style = tkFont.Font(family = font_family, size = font_size["large"], weight = "normal")
    combobox = ttk.Combobox(parent, value = options, font = font_style,state = "readonly")
    return combobox

class MALAS:
    def __init__(self, parent):
        global path_list
        path_list = []
        self.name_list = Listbox(parent, selectmode=SINGLE, width=30,highlightcolor='yellow')

    def loadfile(self):
        plist = list(filedialog.askopenfilenames(title="selectionner des fichiers", filetypes=[('RAD', '*.rad')]))
        if plist:
            for path in plist:
                name=os.path.basename(path.strip('.rad'))
                path_list.append(path)
                self.name_list.insert(END, name)

            self.name_list.pack(side="bottom")

    def showfile(self):
        if self.name_list.curselection():
            global filepath
            filepath=str(path_list[int(self.name_list.curselection()[0])])
            Traitement(filepath).apply()
            

    def delfile(self):
        if self.name_list.curselection():
            index=self.name_list.curselection()[0]
            self.name_list.delete(self.name_list.curselection())
            del path_list[index]


class MALA:
    def __init__(self, path):
        self.path=path

    def getname(self):
        return str(os.path.basename(self.path.strip('.rad')))

    def getrd3(self):
        global rd3data
        with open(self.path.strip('.rad')+".rd3", mode='rb') as rd3data:
            rd3data=rd3data.read()
        rd3=struct.unpack("h"*((len(rd3data))//2), rd3data)
        rd3=np.reshape(rd3,(RAD(self.path).getTraces(),RAD(self.path).getSamples())) 
        rd3=np.transpose(rd3)
        return rd3

class RAD:
    def __init__(self, path):
        self.path=path

    def getinfo(self):
        rad=open(self.path,"r")
        rad1=rad.readlines()
        rad.close()
        parametres={}
        for line in rad1:
            line=line.strip('\n')
            line=line.split(':')
            parametres[str(line[0])]=line[1]
        return parametres

    def getTraces(self):
        return int(self.getinfo()["LAST TRACE"])

    def getSamples(self):
        return int(self.getinfo()["SAMPLES"]) 

    def getTimewindow(self):
        return float(self.getinfo()["TIMEWINDOW"]) 

    def getDx(self):
        return float(self.getinfo()["DISTANCE INTERVAL"])

    def getDt(self):
        return self.getTimewindow()/self.getSamples()

    def listsamples(self):
        return [i for i in range(self.getSamples())]


class Traitement:
    def __init__(self, path):
        self.path=path
        self.rd3_mod=MALA(self.path).getrd3()
        self.listsamples_mod=RAD(self.path).listsamples()

    def plot(self):
        Radargramme(self.path).plot(self.rd3_mod, self.listsamples_mod)
        Scope(self.path).plot(self.rd3_mod,self.listsamples_mod, param.scope)

    def conversion(self, u: float, unit_1, unit_2):
        if u==0:
            return 0
        if u==None:
            return None
        if unit_1==unit_2:
            return u
        if unit_1=="Samples" and unit_2=="Temps (ns)":
            return round(u*RAD(filepath).getDt())
        elif unit_1=="Temps (ns)" and unit_2=="Samples":
            return round(u/RAD(filepath).getDt())
        elif unit_1=="Temps (ns)" and unit_2=="Distance (m)":
            return round((u*1e-9*param.c)/(2*(param.epsilon**0.5)), 2)
        elif unit_1=="Samples" and unit_2=="Distance (m)":
            return round((u*RAD(filepath).getDt()*1e-9*param.c)/(2*(param.epsilon**0.5)), 2)
        elif unit_1=="Distance (m)" and unit_2=="Temps (ns)":
            return round((u*2*(param.epsilon**0.5))/(1e-9*param.c))
        elif unit_1=="Distance (m)" and unit_2=="Samples":
            return round((u*2*(param.epsilon**0.5))/(RAD(filepath).getDt()*1e-9*param.c))
        elif unit_1=="Traces" and unit_2=="Distance (m)":
            return u*RAD(filepath).getDx()
        elif unit_1=="Distance (m)" and unit_2=="Traces":
            return round(u/RAD(filepath).getDx())

    def clear_entry(self, name):
        for w in [".png",".txt",".csv"," "]:
            if w in name:
                name=name.replace(w,"")
        return name

    def timecut(self, start, end):
        if start==None:
            start=0
        else: start=self.conversion(start, param.y_unit, "Samples")
        if end==None:
            end=RAD(filepath).getSamples()
        else: end=self.conversion(end, param.y_unit, "Samples")
        
        self.listsamples_mod=self.listsamples_mod[start:end]
        self.rd3_mod=self.rd3_mod[start:end]

    def gainlin(self, a, t0):
        t0=self.conversion(t0, param.y_unit, "Samples")
        samples, Traces=self.rd3_mod.shape
        fgain=np.ones(samples)
        fgain[t0:]=[a*(x-t0)+1 for x in self.listsamples_mod[t0:]]
        for trace in range(Traces):
            self.rd3_mod[:, trace] = self.rd3_mod[:, trace] * np.array(fgain).astype(dtype=self.rd3_mod.dtype)

    def gainstatique(self, c):
        self.rd3_mod=np.multiply(self.rd3_mod, float(c))

    def gainexp(self, a, t0):
        t0=self.conversion(t0, param.y_unit, "Samples")
        b=np.log(1/a)/t0
        samples, Traces=self.rd3_mod.shape
        fgain=np.ones(samples)
        fgain=[(a*(np.exp(b*(x)) - 1) + 1) for x in self.listsamples_mod]
        for trace in range(Traces):
            self.rd3_mod[:, trace] = self.rd3_mod[:, trace] * np.array(fgain).astype(dtype=self.rd3_mod.dtype)

    def sub_mean(self):
        if param.move_avg==None:
            mean_tr = np.mean(self.rd3_mod, axis=1)
            ns, ntr = self.rd3_mod.shape
            for n in range(ntr):
                self.rd3_mod[:,n] = self.rd3_mod[:,n] - mean_tr
        else:
            start=param.move_avg
            end=RAD(filepath).getTraces()-param.move_avg
            ls=np.arange(start, end, 1)
            for n in ls:
                mean_tr = np.mean(self.rd3_mod[:, int(n-param.move_avg):int(n+param.move_avg)], axis=1)
                self.rd3_mod[:,int(n)] = self.rd3_mod[:,int(n)] - mean_tr
            mean_l = np.mean(self.rd3_mod[:, 0:int(start)], axis=1)
            mean_r = np.mean(self.rd3_mod[:, int(end):int(RAD(filepath).getTraces())], axis=1)
            for n in np.arange(0, start, 1):
                self.rd3_mod[:,int(n)] = self.rd3_mod[:,int(n)] - mean_l
            for n in np.arange(end, RAD(filepath).getTraces(), 1):
                self.rd3_mod[:,int(n)] = self.rd3_mod[:,int(n)] - mean_r

    def dc_substraction(self, start, end):
        if start==None:
            start=0
        else: start=self.conversion(start, param.y_unit, "Samples")
        if end==None:
            end=RAD(filepath).getSamples()
        else: end=self.conversion(end, param.y_unit, "Samples")

        self.rd3_mod= self.rd3_mod - np.mean(abs(self.rd3_mod[start:end]))

    def dc_removal(self, start, end):
        if start==None:
            start=0
        else: start=self.conversion(start, param.y_unit, "Samples")
        if end==None:
            end=RAD(filepath).getSamples()
        else: end=self.conversion(end, param.y_unit, "Samples")

        samples, Traces = self.rd3_mod.shape
        mean_s = np.mean(self.rd3_mod, axis=0)
        self.rd3_mod=self.rd3_mod-mean_s
				
    def save_csv(self, name, type):
        if type=="Brute":
            rd3=MALA(filepath).getrd3()
        else: rd3=self.rd3_mod
        name=self.clear_entry(name)
        df=pd.DataFrame(rd3) 
        df.to_csv('/Volumes/GoogleDrive/Drive partagés/nablaPy/'+str(name)+'.csv') 

    def save_data(self, name):
        name=self.clear_entry(name)
        newRD3path=open('/Volumes/GoogleDrive/Drive partagés/nablaPy/'+str(name)+'.rd3',"wb") 
        newRD3=struct.pack("h"*((len(rd3data))//2), np.transpose(self.rd3_mod))

        newRAD=open('/Volumes/GoogleDrive/Drive partagés/nablaPy/'+MALA(self.path).getname()+"_edit.rad","x")
        rad=open(self.path,"r")
        for line in rad.readlines():
            newRAD.write(line)
        rad.close()
            
    def save_png(self, name, png):
        if png=="1 seul":
            self.save_png_2(name, filepath)
        else:
            for k in range(len(path_list)):
                label=MALA(path_list[k]).getname()
                self.save_png_2(label, path_list[k])
    
    def save_png_2(self, name, path):
        rd3, samples= Traitement(path).apply()
        figu=plt.figure(figsize=(6,3))
        radargramme=figu.add_subplot(1,1,1)
        radargramme.xaxis.set_ticks_position("top")
        radargramme.xaxis.set_label_position('top')
        plt.xticks(fontsize= 5)
        plt.yticks(fontsize= 5)
        plt.locator_params(axis='y', nbins=param.nbTick)
        plt.locator_params(axis='x', nbins=param.nbTick)
        img=radargramme.imshow(rd3, cmap=param.color, interpolation=param.interpolation, aspect='auto', extent=[0,Graph(path).X_list()[-1],Graph(path).Y_list(samples)[-1],0])
        plt.ylabel(param.y_unit, fontsize=6)
        plt.xlabel(param.x_unit, fontsize=6)
        img.set_clim(-param.plot_scale,param.plot_scale)
        figu.savefig(str(name)+'.png', dpi=1000, format='png', bbox_inches='tight')
        plt.close(figu)


    def invert(self):
        self.rd3_mod=np.fliplr(self.rd3_mod)

    def save_param(self, name):
        name=self.clear_entry(name)
        with open(name+'.nablapy', 'wb') as file:
            pickle.dump(vars(param), file)

    def open_param(self):
        path=filedialog.askopenfilename(title="selectionner un fichier", filetypes=[('NABLAPY', '*.nablapy')])
        if path:
            with open(path, 'rb') as file:
                param_loaded = pickle.load(file)
        param.update(param_loaded)
        
    def g_max(self):
        self.rd3_mod=np.clip(self.rd3_mod, -param.g_max, param.g_max)

    def cut_freq(self, fc):
        dt=RAD(filepath).getDt()
        index = int(fc*dt*1e-9)
        fftdata = np.fft.fft2(self.rd3_mod)
        
        for trace in range(RAD(filepath).getTraces()):
            fit = np.diff(fftdata.real[index:index+2, trace]) * [range(index)]
            fftdata.real[:index, trace] = fit

        rd3 = np.fft.ifft2(fftdata)
        self.rd3_mod = rd3.real

    def trace_reduction(self, n):
        if n>100:
            pass
        if n<0:
            pass
        p=round((n*RAD(filepath).getTraces())/100)
        L=np.linspace(0,RAD(filepath).getTraces()-1, p, dtype = int)
        self.rd3_mod=np.delete(self.rd3_mod,L,1)


        

    def apply(self):
        if param.trace_reduction==True: 
            self.trace_reduction(param.traces)
        if param.dc_removal==True: 
            self.dc_removal(param.start_dc_r, param.end_dc_r)
        if param.cut_freq==True: 
            self.cut_freq(param.cut_fc) 
        if param.sub_mean==True: 
            self.sub_mean()
        if param.dc_substraction==True: 
            self.dc_substraction(param.start_dc_s, param.end_dc_s) 
        if param.g_line==True: 
            self.gainlin(param.a_line, param.b_line)
        if param.g_exp==True: 
            self.gainexp(param.a_exp, param.b_exp)
        if param.g_cst==True: 
            self.gainstatique(param.c_gain)
        if param.invert==True: 
            self.invert()
    
        self.g_max()
        self.timecut(param.start, param.end)
        self.plot()
        return self.rd3_mod, self.listsamples_mod


class Parametres:
    
    start=0
    end=None
    y_unit="Temps (ns)"
    x_unit="Traces"
    scope=1
    color='Greys'
    epsilon=8
    c=299792458
    g_max=25000
    plot_scale=25000
    c_gain=1
    a_exp=1
    b_exp=0
    a_line=1
    b_line=0
    g_cst=False
    g_exp=False
    g_line=False
    sub_mean=False
    dc_removal=False
    dc_substraction=False
    start_dc_r=0
    end_dc_r=None
    start_dc_s=0
    end_dc_s=None
    nbTick=20
    invert=False
    move_avg=None
    cut_freq=False
    cut_fc=None
    trace_reduction=False
    traces=100
    interpolation="nearest"
    slice_color="jet"
    start_mean=20
    end_mean=35
    interline=50
    invert_trace=False
    slice_unit="Samples"
    grid=None
    max_auto=False
    slice_cut_start=0
    slice_cut_end=None
    slice_cut=False
    slice_bar=None
    grid_slice=False
    hauteur=9

    def update(self,newdata):
        for key,value in newdata.items():
            setattr(self,key,value)

class Graph:
    def __init__(self, path):
        self.path=path
        self.nbTraces=RAD(self.path).getTraces()

    def Y_list(self, listsamples):
        if param.y_unit == "Temps (ns)":
            return np.multiply(listsamples,RAD(self.path).getDt())
        if param.y_unit == "Samples":
            self.Y_label="Samples"
            return listsamples
        if param.y_unit=="Distance (m)":
            return np.multiply(listsamples, RAD(self.path).getDt()*1e-9*param.c/(2*(param.epsilon**0.5)))

    def X_list(self):
        if param.x_unit=="Distance (m)":
            dx=RAD(self.path).getDx()
            return np.multiply(np.arange(0, self.nbTraces, 1),dx)
        if param.x_unit=="Traces":
            return np.arange(0, self.nbTraces, 1)

    def close(self):
            plt.close()



class Radargramme(Graph):
    def __init__(self, path):
        super().__init__(path)
        self.figu=plt.figure(figsize=(9,param.hauteur))
        self.radargramme=self.figu.add_subplot(1,1,1)
        self.figu.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.06)
        self.radargramme.xaxis.set_ticks_position("top")
        plt.locator_params(axis='y', nbins=param.nbTick)
        plt.locator_params(axis='x', nbins=param.nbTick)
        plt.title(str(MALA(self.path).getname()), fontsize=6, y=-0.01)
        plt.xticks(fontsize= 8)
        plt.yticks(fontsize= 8)
        self.canvas = FigureCanvasTkAgg(self.figu, master=tabRadar)
        self.canvas.get_tk_widget().grid(row=0,column=0)
        self.canvas.draw()
        

    def plot(self, rd3, listsamples):
        img=self.radargramme.imshow(rd3, interpolation=param.interpolation, aspect = 'auto', cmap=param.color, extent=[0,self.X_list()[-1],self.Y_list(listsamples)[-1],0])
        img.set_clim(-param.plot_scale,param.plot_scale)
        plt.axvline(Traitement(filepath).conversion(param.scope,"Traces",param.x_unit), color='r', linewidth=0.8)
        if param.grid==True:
            plt.grid(axis=Y, linewidth = 0.3, color="black", linestyle='-')
        if param.slice_bar==True:
            start=Traitement(filepath).conversion(param.start_mean, "Temps (ns)", param.y_unit)-param.start
            end=Traitement(filepath).conversion(param.end_mean, "Temps (ns)", param.y_unit)-param.start
            print(start, end)
            plt.axhline(start, color='g', linewidth=0.8)
            plt.axhline(end, color='g', linewidth=0.8)
        self.canvas = FigureCanvasTkAgg(self.figu, master=tabRadar)
        self.canvas.get_tk_widget().grid(row=0,column=0)
        self.canvas.draw()
        plt.close(self.figu)

    def canvas_del(self):
        self.canvas.delete("all")
       

class Scope(Graph):
    def __init__(self, path):
        super().__init__(path)
        self.figu=plt.figure(figsize=(1.5,param.hauteur))
        self.scope=self.figu.add_subplot(1,1,1)
        self.scope.invert_yaxis()
        self.scope.axes.get_xaxis().set_visible(False)
        self.scope.axes.get_yaxis().set_visible(False)
        self.figu.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.06)
        plt.title('scope n°: '+str(param.scope), fontsize=6)
        plt.margins(0)
        plt.axvline(x=0, color="g", linewidth=0.3)
        self.canvas.get_tk_widget().grid(row=0,column=1)
        self.canvas.draw()
        plt.close(self.figu)
        

    def plot(self, rd3, listsamples, n):
        img=self.scope.plot([line[n] for line in rd3], self.Y_list(listsamples), color='black', linewidth=0.3)
        self.canvas = FigureCanvasTkAgg(self.figu, master=tabRadar)
        self.canvas.get_tk_widget().grid(row=0,column=1)
        self.canvas.draw()
        plt.close(self.figu)

class Slice:
    def __init__(self):
        self.figu=plt.figure(figsize=(7,7))
        self.slice=self.figu.add_subplot(1,1,1)
        self.figu.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.06)
        self.slice.xaxis.set_ticks_position("top")
        plt.xticks(fontsize= 8)
        plt.yticks(fontsize= 8)

    def X_list(self, slice):
        nbTraces=len(slice[0])
        dx=RAD(path_list[0]).getDx()
        return np.multiply(np.arange(0, nbTraces, 1),dx)


    def Y_list(self, slice):
        dy=param.interline/100
        n=len(slice)
        return np.arange(0,n*dy, dy)

    def plot(self, slice):
        img=self.slice.imshow(slice ,origin='lower', interpolation='gaussian', cmap=param.slice_color, aspect='auto', extent=[0, self.X_list(slice)[-1], self.Y_list(slice)[-1],0 ])
        self.figu.colorbar(img, orientation="horizontal",aspect=50)
        plt.locator_params(axis='x', nbins=param.nbTick)
        plt.locator_params(axis='y', nbins=param.nbTick)
        if param.max_auto==True:
            max=slice.max()
            img.set_clim(0,max)
        else:
            img.set_clim(0,25000)
        if param.grid_slice==True:
            plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        canvas = FigureCanvasTkAgg(self.figu, master=plot_frame)
        canvas.get_tk_widget().grid(row=0, column=0)
        canvas.draw()
        plt.close(self.figu)

class Traitement_slice:
    def __init__(self, pathlist):
        self.pathlist=pathlist
        self.datalist=[]
        for path in self.pathlist:
            self.datalist.append(MALA(path).getrd3())

    def plot(self):
        Slice().plot(self.slice_mean)

    def sub_mean(self):
        for rd3 in self.datalist:
            if param.move_avg==None:
                mean_tr = np.mean(self.rd3, axis=1)
                ns, ntr = self.rd3.shape
                for n in range(ntr):
                    self.rd3[:,n] = self.rd3[:,n] - mean_tr
            else:
                start=param.move_avg
                end=RAD(filepath).getTraces()-param.move_avg
                ls=np.arange(start, end, 1)
                for n in ls:
                    mean_tr = np.mean(self.rd3[:, int(n-param.move_avg):int(n+param.move_avg)], axis=1)
                    self.rd3[:,int(n)] = self.rd3[:,int(n)] - mean_tr
                mean_l = np.mean(self.rd3[:, 0:int(start)], axis=1)
                mean_r = np.mean(self.rd3[:, int(end):int(RAD(filepath).getTraces())], axis=1)
                for n in np.arange(0, start, 1):
                    self.rd3[:,int(n)] = self.rd3[:,int(n)] - mean_l
                for n in np.arange(end, RAD(filepath).getTraces(), 1):
                    self.rd3[:,int(n)] = self.rd3[:,int(n)] - mean_r


    def gainstatique(self, c):
        for rd3 in self.datalist:
            rd3=np.multiply(rd3, float(c))

    def gainlin(self, a, t0):
        t0=Traitement(filepath).conversion(t0, param.y_unit, "Samples")
        for rd3 in self.datalist:
            samples, Traces=rd3.shape
            L=[k for k in range(samples)]
            fgain=np.ones(samples)
            fgain[t0:]=[a*(x-t0)+1 for x in L[t0:]]
            for trace in range(Traces):
                rd3[:, trace] = rd3[:, trace] * np.array(fgain).astype(dtype=rd3.dtype)
        
    def gainexp(self, a, t0):
        t0=Traitement(filepath).conversion(t0, param.y_unit, "Samples")
        for rd3 in self.datalist:
            b=np.log(1/a)/t0
            samples, Traces=rd3.shape
            fgain=np.ones(samples)
            L=[k for k in range(samples)]
            fgain=[(a*(np.exp(b*(x)) - 1) + 1) for x in L]
            for trace in range(Traces):
                rd3[:, trace] = rd3[:, trace] * np.array(fgain).astype(dtype=rd3.dtype)

    def get_slice(self, n):
        self.slice=[]
        for k in range(len(self.datalist)):
            a=self.datalist[k][n]
            self.slice.append(list(a))

        return self.filler(self.slice)

    def filler(self, slice):
        maxlen=self.find_max_list(slice)
        for line in slice:
            m=maxlen-len(line)
            for k in range(m):
                line.insert(0,0)
        return slice

    def find_max_list(self, list):
        list_len = [len(i) for i in list]
        return max(list_len)

    def meaner(self, t1, t2):
        start=self.conversion(t1, "Temps (ns)", "Samples")
        end=self.conversion(t2, "Temps (ns)", "Samples")
        self.slice_mean=np.mean( np.array([np.absolute(self.get_slice(k)) for k in range(start, end)]), axis=0 )

    def invert_trace(self):
        for k in range(0, len(self.datalist), 2):
            self.datalist[k]=np.fliplr(self.datalist[k])

    def slice_cut(self, d1, d2):
        start=self.conversion(d1, "Distance (m)", "Traces")
        end=self.conversion(d2, "Distance (m)", "Traces")
        m=len(self.slice_mean[0])-end
        L=[k for k in range(start)]+[k for k in range(m, len(self.slice_mean[0]))]
        self.slice_mean=np.delete(self.slice_mean, L, axis=1)


    def apply(self):
        if param.invert_trace==True:
            self.invert_trace()
        if param.sub_mean==True: 
            self.sub_mean()
        if param.g_line==True: 
            self.gainlin(param.a_line, param.b_line)
        if param.g_exp==True: 
            self.gainexp(param.a_exp, param.b_exp)
        if param.g_cst==True: 
            self.gainstatique(param.c_gain)
        self.meaner(param.start_mean, param.end_mean)
        if param.slice_cut==True:
            self.slice_cut(param.slice_cut_start, param.slice_cut_end)
        self.plot()
        return self.slice_mean

    def save(self, name):
        slice_mean=self.apply()
        figu=plt.figure(figsize=(4,4))
        slice=figu.add_subplot(1,1,1)
        plt.xticks(fontsize= 7)
        plt.yticks(fontsize= 7)
        img=slice.imshow(slice_mean ,origin='lower', interpolation='gaussian', cmap=param.slice_color, aspect='auto', extent=[0, Slice().X_list(slice_mean)[-1], Slice().Y_list(slice_mean)[-1],0 ])
        figu.colorbar(img, orientation="horizontal",aspect=50)
        if param.max_auto==True:
            max=slice.max()
            img.set_clim(0,max)
        else:
            img.set_clim(0,25000)
        if param.grid_slice==True:
            plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        figu.savefig(str(name)+'.png', dpi=1000, format='png', bbox_inches='tight')
        plt.close(figu)   

    def conversion(self, u: float, unit_1, unit_2):
        if u==0:
            return 0
        if u==None:
            return None
        if unit_1==unit_2:
            return int(u)
        if unit_1=="Samples" and unit_2=="Temps (ns)":
            return round(u*RAD(path_list[0]).getDt())
        elif unit_1=="Temps (ns)" and unit_2=="Samples":
            return round(u/RAD(path_list[0]).getDt())
        elif unit_1=="Temps (ns)" and unit_2=="Distance (m)":
            return round((u*1e-9*param.c)/(2*(param.epsilon**0.5)), 2)
        elif unit_1=="Samples" and unit_2=="Distance (m)":
            return round((u*RAD(path_list[0]).getDt()*param.c)/(2*(param.epsilon**0.5)), 2)
        elif unit_1=="Distance (m)" and unit_2=="Temps (ns)":
            return round((u*2*(param.epsilon**0.5))/(1e-9*param.c))
        elif unit_1=="Distance (m)" and unit_2=="Samples":
            return round((u*2*(param.epsilon**0.5))/(RAD(path_list[0]).getDt()*1e-9*param.c))
        elif unit_1=="Distance (m)" and unit_2=="Traces":
            return round(u/RAD(path_list[0]).getDx())
      



class Toolframe:
    def __init__(self, parent):
        self.tool_frame_label(parent)
    
    def tool_frame_label(self, parent):

        tool_frame = label_frame(parent, "Parametres")
        tool_frame.configure(width=300)
        tool_frame.pack(side=LEFT,fill=Y, expand=0, padx=5, pady=5)
        tool_frame.grid_propagate(0)
        tool_frame.pack_propagate(0)
        self.fill_tool_frame(tool_frame)


    def fill_tool_frame(self, parent):
        global param
        global filepath
        filepath=None
        param=Parametres()
        ft=font.Font(size=12)
        self.tab=False

        #GESTION FICHIER
        fileFrame=Frame(parent)
        fileFrame.pack(side="top", fill = BOTH)
        files=MALAS(fileFrame)

        fileButtonFrame=Frame(fileFrame)
        fileButtonFrame.pack(side='top', fill = BOTH)

        importer=Button(fileButtonFrame, text="Importer", command=lambda:files.loadfile(), font=ft, width=7)
        importer.grid(row=0, column=0, sticky=W+E)

        afficher=Button(fileButtonFrame, text="Afficher", command=lambda:[files.showfile(), self.fill_tab(), self.update()], font=ft, width=7)
        afficher.grid(row=0, column=1, sticky=W+E)

        supprimer=Button(fileButtonFrame, text="Supprimer", command=lambda:files.delfile(), font=ft, width=7)
        supprimer.grid(row=0, column=2, sticky=W+E)

        Label(parent, text="").pack(side='top', fill = BOTH)
        
        #GESTION UNITE, COULEUR
        self.y_unitFrame=Frame(parent)
        self.y_unitFrame.pack(side='top', fill = BOTH)
        self.y_unitbox=combobox(self.y_unitFrame, ["Temps (ns)","Samples","Distance (m)"])
        self.y_unitbox.set(param.y_unit)
        unitLabel=Label(self.y_unitFrame, text="Unité en ordonnée", font=ft)
        unitLabel.grid(row=0, column=0, sticky=W+E)
        self.y_unitbox.grid(row=0, column=1, sticky=W+E)

        self.x_unitbox=combobox(self.y_unitFrame, ["Traces","Distance (m)"])
        self.x_unitbox.set(param.x_unit)
        x_unitLabel=Label(self.y_unitFrame, text="Unité en abscisse", font=ft)
        x_unitLabel.grid(row=1, column=0, sticky=W+E)
        self.x_unitbox.grid(row=1, column=1, sticky=W+E)

        color_label=Label(self.y_unitFrame, text="Couleur").grid(row=2, column=0, stick=W+E)
        self.colorbox=combobox(self.y_unitFrame, ["seismic","Greys", "jet"])
        self.colorbox.set(param.color)
        self.colorbox.grid(row=2, column=1, sticky=W)

        Label(parent, text="").pack(side='top', fill = BOTH)

        self.scrollFrame=ttk.Frame(parent)
        self.scrollFrame.pack(side='top', fill = BOTH)
        

        #GESTION ONGLETS
        tabControl=ttk.Notebook(parent)
        self.tabBasic=ttk.Frame(tabControl)
        self.tabFilter=ttk.Frame(tabControl)
        self.tabConstant=ttk.Frame(tabControl)
        self.tabSave=ttk.Frame(tabControl)
        tabControl.add(self.tabBasic, text="Basique")
        tabControl.add(self.tabFilter, text="Filtres")
        tabControl.add(self.tabConstant, text="Info")
        tabControl.add(self.tabSave, text="Save")
        tabControl.pack(side="top")
        
        apply=Button(parent, text="Appliquer", command=lambda: [self.update(), Traitement(filepath).apply()] )
        apply.pack(side="bottom", fill="both")

    def scroll_bar(self, parent):
        Label(parent, text="Scope").grid(row=3, column=0, sticky=W+E)
        self.scope_scale=Scale(parent, from_=1, to=RAD(filepath).getTraces(), orient=HORIZONTAL)
        self.scope_scale.grid(row=3, column=1, sticky=W+E)
        self.scope_scale.set(param.scope)


    def fill_tab(self):
        if self.tab==False:
            if filepath != None:
                self.fill_tabBasic(self.tabBasic)
                self.fill_tabConstant(self.tabConstant)
                self.fill_tabSave(self.tabSave)
                self.fill_tabFilter(self.tabFilter)
                self.scroll_bar(self.y_unitFrame)
                self.tab=True

    def fill_tabSave(self, parent):
        Label(parent, text="Sauvegarde format png").grid(row=0, columnspan=2, sticky=W+E)
        name_label=label(parent, "Nom fichier")
        name_label.grid(row=1, column=0, sticky=W+E)
        name=entry(parent)
        name.insert(END, str(MALA(filepath).getname()))
        name.grid(row=1, column=1, sticky=W+E)
        pngbox=combobox(parent, ["1 seul","Tout"])
        pngbox.current(0)
        pngbox.grid(row=2, columnspan=2, sticky=W+E)
        savepng_button=Button(parent, text="sauvegarder", command=lambda: Traitement(filepath).save_png(name.get(), pngbox.get())).grid(row=3, columnspan=2, sticky=W+E)
        
        ttk.Separator(parent).grid(row=4, columnspan=2, sticky=W+E, pady=5)

        Label(parent, text="Sauvegarder CSV").grid(row=5, columnspan=2, stick=W+E)
        name_label_csv=label(parent, "Nom fichier")
        name_label_csv.grid(row=6, column=0, sticky=W+E)
        name_csv=entry(parent)
        name_csv.insert(END, str(MALA(filepath).getname()))
        name_csv.grid(row=6, column=1, sticky=W+E)
        csvbox=combobox(parent, ["Brute","Traité"])
        csvbox.current(0)
        csvbox.grid(row=7, columnspan=2, sticky=W+E)
        savecsv_button=Button(parent, text="sauvegarder", command=lambda: Traitement(filepath).save_csv(name_csv.get(), csvbox.get())).grid(row=8, columnspan=2, sticky=W+E)
        
        ttk.Separator(parent).grid(row=9, columnspan=2, sticky=W+E, pady=5)
        
        Label(parent, text="Sauvegarder Paramètres").grid(row=10, columnspan=2, stick=W+E)
        name_label_param=label(parent, "Nom fichier")
        name_label_param.grid(row=11, column=0, sticky=W+E)
        name_param=entry(parent)
        name_param.insert(END, str(MALA(filepath).getname()))
        name_param.grid(row=11, column=1, sticky=W+E)
        saveparam_button=Button(parent, text="sauvegarder", command=lambda: Traitement(filepath).save_param(name_param.get())).grid(row=12, columnspan=2, sticky=W+E)


    def fill_tabFilter(self, parent):
        Label(parent, text="Substract mean").grid(row=0, column=0, sticky=W+E)
        self.sub_mean=tk.IntVar()
        Checkbutton(parent, onvalue=1, offvalue=0, variable=self.sub_mean).grid(row=0, column=1, stick=W+E)
        Label(parent, text="Moyenne glissante").grid(row=1, column=0, sticky=W+E)
        self.move_avg=entry(parent)
        self.move_avg.grid(row=1, column=1, sticky=W+E)
        self.move_avg.insert(END, str(param.move_avg))

        ttk.Separator(parent).grid(row=2, columnspan=2, sticky=W+E, pady=5)

        Label(parent, text="Dewow").grid(row=3, column=0, sticky=W+E)
        self.dc_removal=tk.IntVar()
        Checkbutton(parent, onvalue=1, offvalue=0, variable=self.dc_removal).grid(row=3, column=1, stick=W+E)
        self.start_dc_r=entry(parent)
        self.end_dc_r=entry(parent)
        self.start_dc_r.grid(row=4, column=0, sticky=W+E)
        self.end_dc_r.grid(row=4, column=1, sticky=W+E)
        self.start_dc_r.insert(END, str(param.start_dc_r))
        self.end_dc_r.insert(END, str(param.end_dc_r))

        ttk.Separator(parent).grid(row=5, columnspan=2, sticky=W+E, pady=5)

        Label(parent, text="Inverser le sens").grid(row=6, column=0, sticky=W+E)
        self.invert=tk.IntVar()
        Checkbutton(parent, onvalue=1, offvalue=0, variable=self.invert).grid(row=6, column=1, stick=W+E)
        
        ttk.Separator(parent).grid(row=7, columnspan=2, sticky=W+E, pady=5)

        Label(parent, text="Passe bande").grid(row=8, column=0, sticky=W+E, pady=5)
        self.cut_freq=tk.IntVar()
        Checkbutton(parent, onvalue=1, offvalue=0, variable=self.cut_freq).grid(row=8, column=1, stick=W+E)
        Label(parent, text="fréquence").grid(row=9, column=0, sticky=W+E, pady=5)
        self.cut_fc=entry(parent)
        self.cut_fc.insert(END, str(param.cut_fc))
        self.cut_fc.grid(row=9, column=1, sticky=W+E, pady=5)

        ttk.Separator(parent).grid(row=10, columnspan=2, sticky=W+E, pady=5)

        Label(parent, text="Trace reduction").grid(row=11, column=0, sticky=W+E, pady=5)
        self.trace_reduction=tk.IntVar()
        Checkbutton(parent, onvalue=1, offvalue=0, variable=self.trace_reduction).grid(row=11, column=1, stick=W+E)
        Label(parent, text="Traces (%)").grid(row=12, column=0, sticky=W+E, pady=5)
        self.traces=entry(parent)
        self.traces.insert(END, str(param.traces))
        self.traces.grid(row=12, column=1, sticky=W+E, pady=5)


    def fill_tabConstant(self, parent):
        label(parent, "Nombre de samples").grid(row=2, column=0, sticky=W+E)
        label(parent, RAD(filepath).getSamples()).grid(row=2, column=1, sticky=W+E)
        label(parent, "Nombre de Traces").grid(row=3, column=0, sticky=W+E)
        label(parent, RAD(filepath).getTraces()).grid(row=3, column=1, sticky=W+E)
        label(parent, "Pas Distance (m)").grid(row=4, column=0, sticky=W+E)
        label(parent, RAD(filepath).getDx()).grid(row=4, column=1, sticky=W+E)
        label(parent, "dt").grid(row=5, column=0, sticky=W+E)
        label(parent, round(RAD(filepath).getDt(),2)).grid(row=5, column=1, sticky=W+E)
        label(parent, "Epsilon").grid(row=6, column=0, sticky=W+E)
        self.epsilon=entry(parent)
        self.epsilon.grid(row=6, column=1, sticky=W+E)
        self.epsilon.insert(END, str(param.epsilon))

        label(parent, "Gain max").grid(row=7, column=0, sticky=W+E)
        self.g_max=entry(parent)
        self.g_max.grid(row=7, column=1, sticky=W+E)
        self.g_max.insert(END, str(param.g_max))

        label(parent, "Plot Scale").grid(row=8, column=0, sticky=W+E)
        self.plot_scale=entry(parent)
        self.plot_scale.grid(row=8, column=1, sticky=W+E)
        self.plot_scale.insert(END, str(param.plot_scale))

        label(parent, "Nombre tick").grid(row=9, column=0, sticky=W+E)
        self.nbTick=entry(parent)
        self.nbTick.grid(row=9, column=1, sticky=W+E)
        self.nbTick.insert(END, str(param.nbTick))


        label(parent, "Interpolation").grid(row=10, column=0, sticky=W+E)
        self.interpolation=combobox(parent, ["nearest","gaussian","none","bilinear"])
        self.interpolation.set(param.interpolation)
        self.interpolation.grid(row=10, column=1, sticky=W+E)

        label(parent, "Grille").grid(row=11, column=0, sticky=W+E)
        self.grid=tk.IntVar()
        Checkbutton(parent,  onvalue=1, offvalue=0, variable=self.grid).grid(row=11, column=1, stick=W+E)

        label(parent, "Slice").grid(row=12, column=0, sticky=W+E)
        self.slice_bar=tk.IntVar()
        Checkbutton(parent,  onvalue=1, offvalue=0, variable=self.slice_bar).grid(row=12, column=1, stick=W+E)

        label(parent, "Hauteur 0<h<9").grid(row=13, column=0, sticky=W+E)
        self.hauteur=entry(parent)
        self.hauteur.grid(row=13, column=1, sticky=W+E)
        self.hauteur.insert(END, str(param.hauteur))

        label(parent, "Charger paramètres").grid(row=14, column=0, sticky=W+E)
        load_button=Button(parent, text="charger", command=lambda: [Traitement(filepath).open_param(), self.reset_tab(),Traitement(filepath).apply()])
        load_button.grid(row=14, column=1, sticky=W+E)


    def fill_tabBasic(self, parent):
        #time/sample/distance cut
        cut_label=Label(parent, text="Découpage").grid(row=3, columnspan=2)
        t1_label=Label(parent, text="Début "+param.y_unit).grid(row=4, column=0, stick=W+E)
        t2_label=Label(parent, text="Fin "+param.y_unit).grid(row=5, column=0, stick=W+E)
        self.start=entry(parent)
        self.start.insert(END, str(param.start))
        self.end=entry(parent)
        self.end.insert(END, str(param.end))
        self.start.grid(row=4, column=1, stick=W+E)
        self.end.grid(row=5, column=1, stick=W+E)

        ttk.Separator(parent).grid(row=6, columnspan=2, sticky=W+E, pady=5)

        
        ttk.Separator(parent).grid(row=9, columnspan=2, sticky=W+E, pady=5)

        #gain
        Label(parent, text="Gain constant C*x").grid(row=10, column=0)
        self.g_cst=tk.IntVar()
        Checkbutton(parent, onvalue=1, offvalue=0, variable=self.g_cst).grid(row=10, column=1, stick=W+E)
        c_label=Label(parent, text="c").grid(row=11, column=0, stick=W+E)
        self.c=entry(parent)
        self.c.insert(END, str(param.c_gain))
        self.c.grid(row=11, column=1, sticky=W+E)

        Label(parent, text="Gain exp a*exp(b*x)").grid(row=12, column=0, stick=W+E)
        self.g_exp=tk.IntVar()
        Checkbutton(parent, onvalue=1, offvalue=0, variable=self.g_exp).grid(row=12, column=1, stick=W+E)
        aexp_label=Label(parent, text="0<a<1").grid(row=13, column=0, stick=W+E)
        self.a_exp=entry(parent)
        self.a_exp.insert(END, str(param.a_exp))
        self.a_exp.grid(row=13, column=1, sticky=W+E)
        bexp_label=Label(parent, text="t0").grid(row=14, column=0, stick=W+E)
        self.b_exp=entry(parent)
        self.b_exp.insert(END, str(param.b_exp))
        self.b_exp.grid(row=14, column=1, sticky=W+E)

        Label(parent, text="Gain line a*x+b").grid(row=15, column=0, stick=W+E)
        self.g_line=tk.IntVar()
        Checkbutton(parent, onvalue=1, offvalue=0, variable=self.g_line).grid(row=15, column=1, stick=W+E)
        aline_label=Label(parent, text="a").grid(row=16, column=0, stick=W+E)
        self.a_line=entry(parent)
        self.a_line.insert(END, str(param.a_line))
        self.a_line.grid(row=16, column=1, sticky=W+E)
        bline_label=Label(parent, text="t0").grid(row=17, column=0, stick=W+E)
        self.b_line=entry(parent)
        self.b_line.insert(END, str(param.b_line))
        self.b_line.grid(row=17, column=1, sticky=W+E)

    def reset_tab(self): 
        self.x_unitbox.set(param.x_unit)
        self.y_unitbox.set(param.y_unit)
        self.colorbox.set(param.color)
        for tab in [self.tabBasic, self.tabConstant, self.tabFilter, self.tabSave]:
            for widget in tab.winfo_children():
                widget.destroy()
        self.tab=False
        self.fill_tab()
        if param.g_cst==True:
            self.g_cst.set(1)
        else: self.g_cst.set(0)
        if param.g_line==True:
            self.g_line.set(1)
        else: self.g_line.set(0)
        if param.g_exp==True:
            self.g_exp.set(1)
        else: self.g_exp.set(0)
        if param.dc_removal==True:
            self.dc_removal.set(1)
        else: self.dc_removal.set(0)
        if param.sub_mean==True:
            self.sub_mean.set(1)
        else: self.sub_mean.set(0)
        if param.invert==True:
            self.invert.set(1)
        else: self.invert.set(0)
        if param.trace_reduction==True:
            self.trace_reduction.set(1)
        else: self.trace_reduction.set(0)


    def get_start_tc(self):
        if self.isfloat(self.start.get()):
            param.start=float(self.replacedecimal(self.start.get()))
        else: param.start=0
    def get_end_tc(self):
        if self.isfloat(self.end.get()):
            param.end=float(self.replacedecimal(self.end.get()))
        else: param.end=None
    def get_start_dc_r(self):
        if self.isfloat(self.start_dc_r.get()):
            param.start_dc_r=float(self.replacedecimal(self.start_dc_r.get()))
        else: param.start_dc_r=0
    def get_end_dc_r(self):
        if self.isfloat(self.end_dc_r.get()):
            param.end_dc_r=float(self.replacedecimal(self.end_dc_r.get()))
        else: param.end_dc_r=None
    def get_color(self):
        param.color=self.colorbox.get()
    def get_interpolation(self):
        param.interpolation=self.interpolation.get()
    def get_unit(self):
        param.x_unit=self.x_unitbox.get()
        if param.y_unit!=self.y_unitbox.get():
            param.start=Traitement.conversion(self, param.start, param.y_unit,self.y_unitbox.get())
            param.end=Traitement.conversion(self, param.end, param.y_unit,self.y_unitbox.get())
            param.b_line=Traitement.conversion(self, param.b_line, param.y_unit,self.y_unitbox.get())
            param.b_exp=Traitement.conversion(self, param.b_exp, param.y_unit,self.y_unitbox.get())
            param.y_unit=self.y_unitbox.get()
            self.reset_tab()   
    def get_c(self):
        if self.isfloat(self.c.get()):
            param.c_gain=float(self.replacedecimal(self.c.get()))
        else: param.c_gain=1
    def get_a_exp(self):
        if self.isfloat(self.a_exp.get()):
            param.a_exp=float(self.replacedecimal(self.a_exp.get()))
        else: param.a_exp=1
    def get_b_exp(self):
        if self.isfloat(self.b_exp.get()):
            param.b_exp=float(self.replacedecimal(self.b_exp.get()))
        else: param.b_exp=1
    def get_a_line(self):
        if self.isfloat(self.a_line.get()):
            param.a_line=float(self.replacedecimal(self.a_line.get()))
        else: param.a_line=1
    def get_b_line(self):
        if self.isfloat(self.b_line.get()):
            param.b_line=float(self.replacedecimal(self.b_line.get()))
        else: param.b_line=1
    def get_epsilon(self):
        if self.isfloat(self.epsilon.get()):
            param.epsilon=float(self.replacedecimal(self.epsilon.get()))
        else: param.epsilon=8
    def get_move_avg(self):
        if self.isfloat(self.move_avg.get()):
            param.move_avg=float(self.replacedecimal(self.move_avg.get()))
        else: param.move_avg=None
    def get_cut_fc(self):
        if self.isfloat(self.cut_fc.get()):
            param.cut_fc=float(self.replacedecimal(self.cut_fc.get()))
        else: param.cut_fc=None
    def get_g_max(self):
        if self.isfloat(self.g_max.get()):
            param.g_max=float(self.replacedecimal(self.g_max.get()))
        else: param.g_max=25000
    def get_plot_scale(self):
        if self.isfloat(self.plot_scale.get()):
            param.plot_scale=float(self.replacedecimal(self.plot_scale.get()))
        else: param.plot_scale=25000
    def get_nbTick(self):
        if self.isfloat(self.nbTick.get()):
            param.nbTick=float(self.replacedecimal(self.nbTick.get()))
        else: param.nbTick=10
    def get_traces(self):
        if self.isfloat(self.traces.get()):
            param.traces=float(self.replacedecimal(self.traces.get()))
        else: param.traces=100
    def resetframe(self, tab):
        for widget in tab.winfo_children():
            widget.destroy()
        self.fill_tabBasic(self.tabBasic)
    def get_hauteur(self):
        if self.isfloat(self.hauteur.get()):
            param.hauteur=float(self.replacedecimal(self.hauteur.get()))
        else: param.hauteur=9


    def get_scope(self):
        param.scope=int(self.scope_scale.get())
    def get_gain(self):
        if self.g_cst.get()==1:
            param.g_cst=True
        else: param.g_cst=False
        if self.g_exp.get()==1:
            param.g_exp=True
        else: param.g_exp=False
        if self.g_line.get()==1:
            param.g_line=True
        else: param.g_line=False  
    def get_filter(self):
        if self.sub_mean.get()==1:
            param.sub_mean=True
        else: param.sub_mean=False
        if self.dc_removal.get()==1:
            param.dc_removal=True
        else: param.dc_removal=False
        if self.cut_freq.get()==1:
            param.cut_freq=True
        else: param.cut_freq=False
        if self.invert.get()==1:
            param.invert=True
        else: param.invert=False
        if self.trace_reduction.get()==1:
            param.trace_reduction=True
        else: param.trace_reduction=False
        if self.grid.get()==1:
            param.grid=True
        else: param.grid=False
        if self.slice_bar.get()==1:
            param.slice_bar=True
        else: param.slice_bar=False

    def update(self):
        self.get_start_tc()
        self.get_end_tc()
        self.get_start_dc_r()
        self.get_end_dc_r()
        self.get_c()
        self.get_interpolation()
        self.get_a_exp()
        self.get_b_exp()
        self.get_a_line()
        self.get_b_line()
        self.get_epsilon()
        self.get_traces()
        self.get_move_avg()
        self.get_cut_fc()
        self.get_g_max()
        self.get_plot_scale()
        self.get_nbTick()
        self.get_gain()
        self.get_filter()
        self.get_color()
        self.get_scope()
        self.get_unit()
        self.get_hauteur()
        Graph(filepath).close()
    
    def isfloat(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def replacedecimal(self, s):
        if "," in s:
            return s.replace(",",".")
        else: return s

    

class Dataframe:
    def __init__(self, parent):
        self.data_frame_label(parent)        

    def data_frame_label(self, parent):
        data_frame=label_frame(parent, "Visualisation")
        data_frame.pack(side=RIGHT, fill="both", expand=1, padx=5, pady=5)
        data_frame.grid_propagate(0)
        data_frame.pack_propagate(0)
        self.fill_data_frame(data_frame)

    def fill_data_frame(self, parent):
        global tabRadar
        global tabSlice
        tabControl=ttk.Notebook(parent)
        tabRadar=ttk.Frame(tabControl)
        tabSlice=ttk.Frame(tabControl)   
        tabControl.add(tabRadar, text="Radargramme")
        tabControl.add(tabSlice, text="Slice")
        tabControl.pack(side="top")
        self.fill_tab_slice(tabSlice)

    
    def fill_tab_slice(self, parent):
        global plot_frame
        plot_frame=Frame(parent)
        plot_frame.pack(side=LEFT, fill=BOTH)


        self.OptionFrame=Frame(parent)
        self.OptionFrame.pack(side=RIGHT, fill = BOTH)

        color_label=Label(self.OptionFrame, text="Couleur").grid(row=0, column=0, stick=W+E)
        self.colorbox=combobox(self.OptionFrame, ["seismic","Greys","jet","rainbow"])
        self.colorbox.set(param.slice_color)
        self.colorbox.grid(row=0, column=1, sticky=W)

        ttk.Separator(self.OptionFrame).grid(row=1, columnspan=2, sticky=W+E, pady=5)


        Label(self.OptionFrame, text="PNG").grid(row=2, columnspan=2)

        self.name=entry(self.OptionFrame)
        self.name.grid(row=2, column=0, sticky=W+E)

        save=Button(self.OptionFrame, text="sauvegarder", command=lambda: Traitement_slice(path_list).save(self.name.get()))
        save.grid(row=2, column=1, sticky=W+E)

        ttk.Separator(self.OptionFrame).grid(row=3, columnspan=2, sticky=W+E, pady=5)
        
        Label(self.OptionFrame, text="Tranche moyenne").grid(row=4, column=0, sticky=W+E)

    
        self.start_scale=self.scroll_bar(self.OptionFrame, "Début (ns)", param.start_mean, 5)
        self.end_scale=self.scroll_bar(self.OptionFrame, "Fin (ns)", param.end_mean, 6)

        ttk.Separator(self.OptionFrame).grid(row=7, columnspan=2, sticky=W+E, pady=5)
        Label(self.OptionFrame, text="Inter-ligne (cm)").grid(row=8, column=0, sticky=W+E)
        self.interline=entry(self.OptionFrame)
        self.interline.grid(row=8, column=1, sticky=W+E)
        self.interline.insert(END, str(param.interline))

        ttk.Separator(self.OptionFrame).grid(row=9, columnspan=2, sticky=W+E, pady=5)
        Label(self.OptionFrame, text="Inverser 1/2").grid(row=10, column=0, sticky=W+E)
        self.invert_trace=tk.IntVar()
        tk.Checkbutton(self.OptionFrame ,onvalue=1, offvalue=0, variable=self.invert_trace).grid(row=10, column=1, sticky=W+E)

        ttk.Separator(self.OptionFrame).grid(row=11, columnspan=2, sticky=W+E, pady=5)
        Label(self.OptionFrame, text="Recherche max auto").grid(row=12, column=0, sticky=W+E)
        self.max_auto=tk.IntVar()
        tk.Checkbutton(self.OptionFrame ,onvalue=1, offvalue=0, variable=self.max_auto).grid(row=12, column=1, sticky=W+E)

        ttk.Separator(self.OptionFrame).grid(row=13, columnspan=2, sticky=W+E, pady=5)
        Label(self.OptionFrame, text="Découpage").grid(row=14, column=0, sticky=W+E)
        self.slice_cut=tk.IntVar()
        tk.Checkbutton(self.OptionFrame ,onvalue=1, offvalue=0, variable=self.slice_cut).grid(row=14, column=1, sticky=W+E)
        Label(self.OptionFrame, text="Début (m)").grid(row=15, column=0)
        self.slice_cut_start=entry(self.OptionFrame)
        self.slice_cut_start.insert(END, str(param.slice_cut_start))
        self.slice_cut_start.grid(row=15, column=1)
        Label(self.OptionFrame, text="Fin (m)").grid(row=16, column=0)
        self.slice_cut_end=entry(self.OptionFrame)
        self.slice_cut_end.insert(END, str(param.slice_cut_end))
        self.slice_cut_end.grid(row=16, column=1)

        ttk.Separator(self.OptionFrame).grid(row=17, columnspan=2, sticky=W+E, pady=5)
        Label(self.OptionFrame, text="Grille").grid(row=18, column=0, sticky=W+E)
        self.grid_slice=tk.IntVar()
        tk.Checkbutton(self.OptionFrame ,onvalue=1, offvalue=0, variable=self.grid_slice).grid(row=18, column=1, sticky=W+E)

        ttk.Separator(self.OptionFrame).grid(row=20, columnspan=2, sticky=W+E, pady=5)
        apply=Button(self.OptionFrame, text="Tracer", command=lambda: [self.update() ,Traitement_slice(path_list).apply()])
        apply.grid(row=21, columnspan=2, sticky=W+E)

        

    def scroll_bar(self, parent, lab, parame, r):
        end=120
        Label(parent, text=str(lab)).grid(row=r, column=0, sticky=W+E)
        self.scope_scale=Scale(parent, from_=1, to=end, orient=HORIZONTAL)
        self.scope_scale.grid(row=r, column=1, sticky=W+E)
        self.scope_scale.set(parame)
        return self.scope_scale

    def get_cut(self):
        param.start_mean=int(self.start_scale.get())
        param.end_mean=int(self.end_scale.get())
        
    def get_cmap(self):
        param.slice_color=self.colorbox.get()

    def get_interline(self):
        if self.isfloat(self.interline.get()):
            param.interline=float(self.replacedecimal(self.interline.get()))
        else: param.interline=50

    def get_invert_trace(self):
        if self.invert_trace.get()==1:
            param.invert_trace=True
        else: param.invert_trace=False

    def get_max_auto(self):
        if self.max_auto.get()==1:
            param.max_auto=True
        else: param.max_auto=False

    def get_slice_cut(self):
        if self.slice_cut.get()==1:
            param.slice_cut=True
        else: param.slice_cut=False
    
    def get_grid(self):
        if self.grid_slice.get()==1:
            param.grid_slice=True
        else: param.grid_slice=False


    def get_slice_cut_start(self):
        if self.isfloat(self.slice_cut_start.get()):
            param.slice_cut_start=float(self.replacedecimal(self.slice_cut_start.get()))
        else: param.slice_cut_start=0

    def get_slice_cut_end(self):
        if self.isfloat(self.slice_cut_end.get()):
            param.slice_cut_end=float(self.replacedecimal(self.slice_cut_end.get()))
        else: param.slice_cut_end=0

    def update(self):
        self.get_cut()
        self.get_cmap()
        self.get_interline()
        self.get_invert_trace()
        self.get_max_auto()
        self.get_slice_cut_start()
        self.get_slice_cut_end()
        self.get_slice_cut()
        self.get_grid()
 
    def isfloat(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def replacedecimal(self, s):
        if "," in s:
            return s.replace(",",".")
        else: return s

    

        


        



main=tk.Tk()
main.geometry('1500x900')
main.title('nablaPy')
Tframe=Toolframe(main)
Dframe=Dataframe(main)

main.mainloop()