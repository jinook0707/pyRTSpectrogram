# coding: UTF-8

"""
Spectrogram:
Drawing real-time spectrogram 

An open-source software written in Python, programmed and tested in
Mac OSX 10.13.

Jinook Oh, Cognitive Biology department, University of Vienna
September 2019.

Dependency:
    wxPython (4.0), 
    Numpy (1.17), 

------------------------------------------------------------------------
Copyright (C) 2019 Jinook Oh, W. Tecumseh Fitch 
- Contact: jinook.oh@univie.ac.at, tecumseh.fitch@univie.ac.at

This program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your 
option) any later version.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along 
with this program.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
"""

import sys, wave
from os import path, getcwd, mkdir 
from copy import copy

import wx
import wx.lib.scrolledpanel as SPanel 
import numpy as np

import pyListenerLib as PLL 
from fFuncNClasses import GNU_notice, show_msg, getWXFonts, setupStaticText

__version__ = "0.1"
DEBUG = False 

#=======================================================================

class SpectrogramPanel(wx.Panel):
    """ Spectrogram panel to be used in wx.Frame.

    Args:
        parent (wx.Frame): Parent frame.
        pos (tuple): Position of this panel.
        sz (tuple): Size of this panel.
        pl (pyListener): pyListener object.
        name (str): Name of panel.

    Attributes: 
        Each attribute is declared and described at the top section 
        of __init__ before wx.Frame.__init__
    """
    def __init__(self, parent, pos, sz, pl, name='sp_panel'):
        if DEBUG: print("SpectrogramPanel.__init__()")

        ##### beginning of class attributes -----
        self.parent = parent
        self.pos = pos 
        self.sz = sz 
        self.pl = pl # pyListener 
        self.fonts = parent.fonts 
        self.flagShowInfo = True # draw info such as framerate, channel, etc.
        self.timers = {} # contain wxPython's timers for this class
        ##### end of class attributes -----

        wx.Panel.__init__(self, parent, wx.ID_ANY, 
                          name=name, pos=pos, size=sz)
        self.Bind(wx.EVT_PAINT, self.onPaint) # paint event
        self.SetBackgroundColour(wx.Colour('#000000')) 
        
        if int(PLL.INPUT_FRAMES_PER_BLOCK/2) > sz[1]:
            msg = "[WARNING], Spectrogram height, calculated with"
            msg += " INPUT_FRAMES_PER_BLOCK, is larger than height of panel." 
            msg += " Please consider lowering INPUT_BLOCK_TIME in PyListener."
            show_msg(msg) 

    #-------------------------------------------------------------------
   
    def onPaint(self, event):
        """ Painting spectrogram.

        Args: event (wx.Event)

        Returns: None
        """ 
        if DEBUG: print("SpectrogramPanel.onPaint()")
        
        evtObj = event.GetEventObject()
        dc = wx.PaintDC(evtObj)
        dc.SetBackground(wx.Brush('#333333'))
        dc.Clear()
        
        ### draw spectrogram 
        sfci = self.pl.sFragCI
        ad = self.pl.spAD 
        imgArr = np.stack( (ad, ad, ad), axis=2 ) 
        img = wx.ImageFromBuffer(imgArr.shape[1], imgArr.shape[0], imgArr)
        bmp = wx.Bitmap(img) # wx.BitmapFromImage(img)
        dc.DrawBitmap(bmp, 0, 0)

        if self.flagShowInfo:
            dc.SetFont(self.fonts[1])
            texts=[]; coords = []; fg = []; bg = []
            fCol = wx.Colour('#999999')
            bCol = wx.Colour('#000000')
            lbl = "Mic. streaming [ Sample-rate:%i,"%(PLL.RATE)
            lbl += " Channels:%i, Data-type:int16,"%(PLL.CHANNELS)
            lbl += " Input-block-time:%.2f,"%(PLL.INPUT_BLOCK_TIME)
            lbl += " Freq.-resolution:%.2f ]"%(PLL.FREQ_RES)
            texts.append(lbl)
            coords.append( (5, 0) )
            fg.append( fCol )
            bg.append( bCol )
            dc.DrawTextList( texts, coords, fg, bg )

        event.Skip()

        ### if parent has post-function (additional drawing, etc), call it.
        postFunc = getattr(self.parent, "postPaintSP", None)
        if callable(postFunc):
            self.parent.postPaintSP(dc)

    #-------------------------------------------------------------------

#=======================================================================

class PyLSpectrogramFrame(wx.Frame):
    """ PyLSpectrogramFrame is wxPython frame for showing spectrogram.

    Attributes: 
        Each attribute is declared and described at the top section 
        of __init__ before wx.Frame.__init__
    """
    def __init__(self):
        if DEBUG: print("PyLSpectrogramFrame.__init__()")

        ##### beginning of class attributes -----
        w_pos = [0, 25]
        w_sz = [800, 600]
        self.w_pos = w_pos
        self.w_sz = w_sz 
        ### panel information
        pi = {} 
        # top user-interface
        pi["tUI"] = dict(pos=(0, 0), 
                        sz=(w_sz[0], 40), 
                        bgCol="#cccccc", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER) 
        # spectrogram panel
        pi["sp"] = dict(pos=(0, pi["tUI"]["sz"][1]), 
                        sz=(w_sz[0], int(PLL.INPUT_FRAMES_PER_BLOCK/2)), 
                        bgCol="#cccccc", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        self.pi = pi # store panel information
        self.panel = {} # dictionary to put panels
        self.gbs = {} # dictionary to put GridBagSizer
        self.fonts = getWXFonts(initFontSz=8, numFonts=3, fSzInc=2)
        self.timers = {} # contain wxPython's timers for this class
        ### init PyListener class 
        self.pl = PLL.PyListener(self, self) 
        if self.pl.devIdx == []: self.onClose(None)
        ##### end of class attributes -----
        
        # numpy array for spectrogram of sound from mic. 
        self.pl.spAD = np.zeros( 
                        (pi['sp']['sz'][1], pi['sp']['sz'][0]), 
                        dtype=np.uint8 
                               )

        ### init frame
        wx.Frame.__init__(self, None, -1, 
                          "PySpectrogram - v.%s"%(__version__), 
                          pos = w_pos, size = self.w_sz) 
        self.SetBackgroundColour('#333333')

        for pk in pi.keys():
            if pk == 'sp':
                self.panel[pk] = SpectrogramPanel(self, 
                                                  name="sp_panel",
                                                  pos=pi["sp"]["pos"], 
                                                  sz=pi["sp"]["sz"], 
                                                  pl=self.pl)
            else:
                self.panel[pk] = SPanel.ScrolledPanel(
                                                      self, 
                                                      name="%s_panel"%(pk), 
                                                      pos=pi[pk]["pos"], 
                                                      size=pi[pk]["sz"], 
                                                      style=pi[pk]["style"],
                                                     )
                self.panel[pk].SetBackgroundColour(pi[pk]["bgCol"]) 

        ##### beginning of setting up top user interface -----
        bw = 5 # border width for GridBagSizer
        vlSz = (-1, 20) # size of vertical line seprator
        self.gbs["tUI"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        sTxt = setupStaticText(
                                    self.panel["tUI"], 
                                    'Input device: ', 
                                    font=self.fonts[2],
                              )
        self.gbs["tUI"].Add(
                            sTxt, 
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           )
        col += 1
        self.devNames_cho = wx.Choice(
                                        self.panel["tUI"], 
                                        -1, 
                                        choices=self.pl.devNames,
                                     )
        self.gbs["tUI"].Add(
                            self.devNames_cho, 
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           )
        col += 1
        self.gbs["tUI"].Add(
                            wx.StaticLine(
                                            self.panel["tUI"],
                                            -1,
                                            size=vlSz,
                                            style=wx.LI_VERTICAL,
                                         ),
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           ) # vertical line separator
        self.panel["tUI"].SetSizer(self.gbs["tUI"])
        self.gbs["tUI"].Layout()
        self.panel["tUI"].SetupScrolling()
        ##### end of setting up top user interface -----

        self.updateFrameSize()

        ### set up menu
        menuBar = wx.MenuBar()
        pyLSpectrogramMenu = wx.Menu()
        startStopListening = pyLSpectrogramMenu.Append(
                            wx.Window.NewControlId(), 
                            item="Start/Stop listening\tSPACE",
                                            )
        self.Bind(wx.EVT_MENU, 
                  self.startStopListening,
                  startStopListening)
        quit = pyLSpectrogramMenu.Append(
                            wx.Window.NewControlId(), 
                            item="Quit\tCTRL+Q",
                              )
        self.Bind(wx.EVT_MENU, self.onClose, quit)
        menuBar.Append(pyLSpectrogramMenu, "&pyLSpectrogram")
        self.SetMenuBar(menuBar)
        
        ### set up hot keys
        idListen = wx.Window.NewControlId()
        idQuit = wx.Window.NewControlId()
        self.Bind(wx.EVT_MENU,
                  self.startStopListening,
                  id=idListen)
        self.Bind(wx.EVT_MENU, self.onClose, id=idQuit)
        accel_tbl = wx.AcceleratorTable([ 
                                    (wx.ACCEL_NORMAL, wx.WXK_SPACE, idListen),
                                    (wx.ACCEL_CMD,  ord('Q'), idQuit), 
                                        ]) 
        self.SetAcceleratorTable(accel_tbl)

        # set up status-bar
        #self.statusbar = self.CreateStatusBar(1)
        #self.sbTimer = None 
   
        self.Bind(wx.EVT_CLOSE, self.onClose) 
   
    #-------------------------------------------------------------------

    def startStopListening(self, event): 
        """ Start/Stop listening with microphone.

        Args: event (wx.Event)
        
        Returns: None
        """ 
        if DEBUG: print("PyLSpectrogramFrame.startStopListening()")
        
        if self.pl.isListening == False:
        # Currently not listening. Start listening.
            ### set up a timer for draw spectrogram
            self.timers["updateSPTimer"] = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, 
                      self.updateSpectrogram, 
                      self.timers["updateSPTimer"])
            self.timers["updateSPTimer"].Start(5)
            # start a thread to listen 
            self.pl.startContMicListening(self.devNames_cho.GetSelection())
        else:
        # Currently listening. Stop.
            self.stop_listening()

    #-------------------------------------------------------------------
    
    def updateSpectrogram(self, event):
        """ Function periodically called by a timer.
        Call a PyListener function to process mic. audio data,
        then, update visual displays including spectrogram.

        Args: event (wx.Event)
        
        Returns: None
        """
        if DEBUG: print("PyLSpectrogramFrame.updateSpectrogram()")
        
        self.pl.procMicAudioData(flagAnalyze=False) 
        self.panel['sp'].Refresh() # draw spectrogram

    #-------------------------------------------------------------------
    
    def stop_listening(self): 
        """ Stop listening from microphone.

        Args: None
        
        Returns: None
       """ 
        if DEBUG: print("PyListenerFrame.stop_listening()")
        ### end timer
        self.timers["updateSPTimer"].Stop()
        self.timers["updateSPTimer"] = None
        
        self.pl.endContMicListening()
        
        self.panel['sp'].Refresh()

    #-------------------------------------------------------------------
    
    def updateFrameSize(self):
        """ Set window size exactly to self.w_sz without menubar/border/etc.

        Args: None

        Returns: None
        """
        if DEBUG: print("PyListenerFrame.updateFrameSize()")
        m = 10 # margin
        # adjust w_sz height to where spectrogram ends
        self.w_sz[1] = self.pi['sp']['pos'][1]+self.pi['sp']['sz'][1] + m 
        ### set window size exactly to self.w_sz 
        ### without menubar/border/etc.
        _diff = (self.GetSize()[0]-self.GetClientSize()[0], 
                 self.GetSize()[1]-self.GetClientSize()[1])
        _sz = (self.w_sz[0]+_diff[0], self.w_sz[1]+_diff[1])
        self.SetSize(_sz) 
        self.Refresh()
    
    #-------------------------------------------------------------------
    
    def onClose(self, event):
        """ Close this frame.

        Args: event (wx.Event)

        Returns: None
        """
        if DEBUG: print("PyLSpectrogramFrame.onClose()")
        if self.pl.th != None:
            self.stop_listening()
            self.pl.pa.terminate()
            wx.CallLater(10, self.Destroy)
        else:
            self.Destroy()

    #-------------------------------------------------------------------

#=======================================================================

class PyLSpectrogramApp(wx.App):
    """ Initializing pyLSpectrogram app with PyLSpectrogramFrame.

    Attributes:
        frame (wx.Frame): PySepctrogramFrame.
    """
    def OnInit(self):
        if DEBUG: print("PyLSpectrogramApp.OnInit()")
        self.frame = PyLSpectrogramFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#=======================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-w': GNU_notice(1)
        elif sys.argv[1] == '-c': GNU_notice(2)
    else:
        CWD = getcwd()
        GNU_notice(0)
        app = PyLSpectrogramApp(redirect = False)
        app.MainLoop()

