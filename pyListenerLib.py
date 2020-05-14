# coding: UTF-8

"""
PyListener library
An open-source software written in Python for sound comparison. 
Currently, its main functionality is to listen to sound from microphone
and save a recognized sound as WAV file, when it is similar with a 
loaded template sound.

This was programmed and tested in macOS 10.13.

Jinook Oh, Cognitive Biology department, University of Vienna
September 2019.

Dependency:
    wxPython (4.0),
    pyAudio (0.2),
    NumPy (1.17),
    SciPy (1.3),
    Scikit-image (0.15),

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

import queue, wave, struct
from os import path, mkdir, getcwd
from threading import Thread
from time import time
from copy import copy
from glob import glob

import pyaudio
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage.measurements import center_of_mass 
from scipy.signal import correlate
from skimage import filters
from skimage import transform 
#from pyentrp import entropy as ent

from fFuncNClasses import chkFPath, writeFile, get_time_stamp
from fFuncNClasses import receiveDataFromQueue

### Constants (time related contants are in seconds)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SAMPLE_WIDTH = 2 # 2 bytes
SHORT_NORMALIZE = 1.0 / 32768
INPUT_BLOCK_TIME = 0.025
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)
FREQ_RES = RATE/float(INPUT_FRAMES_PER_BLOCK)  # Frequency Resolution

DEBUG = False
CWD = getcwd()

#=======================================================================

class PyListener(object):
    """ Class for getting streaming data from mic., 
        analyze/compare audio data.

        Args:
            parent (): Parent object
            frame (wx.Frame, optional): wxPython frame for display.
            logFile (str, optional): File path of log file.
            debug (None/ bool, optional): Debugging mode or not.

        Attributes:
            Each attribute is described on the line in __init__.
    """ 
    def __init__(self, parent, frame=None, logFile=''):
        if DEBUG: print("PyListener.__init__()")
        self.parent = parent
        self.frame = frame
        if frame == None: self.spWidth = 500
        if path.isdir('log') == False: mkdir('log')
        if logFile == '':
            ts =  get_time_stamp()[:-9] # cut off hh_mm_ss from timestamp
            self.logFile = "log/log_%s.txt"%(ts)
        else:
            self.logFile = logFile

        # parameter keys
        self.pKeys = [
                        'duration', 'summedAmp', 'summedAmpRatio', 
                        'cmInColList', 'centerOfMassX', 'centerOfMassY', 
                        'cmxN', 'cmyN', #'permEnt', 
                        'avgNumDataInCol', 'lowFreqRow', 'lowFreq', 
                        'highFreqRow', 'highFreq', 'distLowRow2HighRow', 
                        #'corr2auto'
                     ]
        # parameters to use in comparison between sound fragment 
        # and template WAV
        self.compParamList = [
                                'duration', 'cmxN', 'cmyN', 
                                'avgNumDataInCol', 'lowFreq', 'highFreq', 
                                'distLowRow2HighRow', 'summedAmpRatio', 
                             ] 
        # parameters with values which don't change 
        # when template folder is changed 
        self.indCPL = [
                        'summedAmpRatio', 
                        #'corr2auto',
                      ]
        # -/+ to min and max margins on analyzed threshold ranges
        self.cplInitMargin = dict(
                        duration_min=0.25, duration_max=1.0, 
                        cmxN_min=0.15, cmxN_max=0.15,
                        cmyN_min=0.15, cmyN_max=0.15,
                        avgNumDataInCol_min=20, avgNumDataInCol_max=20,
                        lowFreq_min=2.0, lowFreq_max=2.0, 
                        highFreq_min=2.0, highFreq_max=2.0,
                        distLowRow2HighRow_min=20, distLowRow2HighRow_max=20
                                 ) 
        # threshold ranges for indCPL
        self.indCPRange = dict(
                                summedAmpRatio_min=0.5, summedAmpRatio_max=3.0,
                                #corr2auto_min=0.3, corr2auto_max=1.0,
                              )
        # labels to appear in UI for threshold ranges
        self.compParamLabel = dict(
                        duration='Duration', 
                        cmxN='CenterOfMass-X (0.0-1.0)', 
                        cmyN='CenterOfMass-Y (0.0-1.0)',
                        #permEnt='Permutation Entropy', 
                        avgNumDataInCol='Avg. Num. Data in a column', 
                        lowFreq='Low frequency', 
                        highFreq='High frequency', 
                        distLowRow2HighRow='Distance betw. low && high rows',
                        summedAmpRatio='Summed amp./Template summed amp.'
                        #corr2auto='Correlation/AutoCorrelation', 
                                  )
         
        self.prefDevStr = ['headset', 'built-in']  # strings to detect 
          # preferred audio input device. Order matters. The 1st item is 
          # the most preferred device.
        self.comp_freq_range = [1000, 20000]  # frequency range 
          # for comparing sound
        self.ampMonDur = 0.5  # time (in seconds) to look back for 
          # average amplitude. should be longer than self.minDur4SF.
        self.minDur4SF = 0.5  # minimum duration (in seconds) for a sound fragment
        self.ampRecLen = int(self.ampMonDur/INPUT_BLOCK_TIME)  # length of 
          # 'amps' list. (for measuring recent record) 
        self.ampThr = 0.005  # amplitude (0.0-1.0) threshold to start a sound 
          # fragment (in recent audio data during self.ampMonDur)
        self.maxDurLowerThr = 0.1  # once amp. goes above threshold, 
          # the program will continute to capture audio data until 
          # amp goes below self.ampThr longer than self.maxDurLowerThr.
        self.acThrTol_templ = 3.0  # tolerance value for 
          # filters.threshold_li function to detect threshold for 
          # auto-contrast. higher this value is, less data points will 
          # appear in spectrogram. 
          # tolerance: Finish the computation when the change in the 
          # threshold in an iteration is less than this value. 
          # By default, this is half the smallest difference 
          # between intensity values in image.
        self.acThrTol_nt = 3.0
        self.rMicData = []  # data read from mic 
        self.spAD = None  # NumPy array to store recent audio data 
          # for drawing spectrogram
        self.tSpAD = None  # NumPy array to store audio data of 
          # selected WAV file 
        self.th = None # thread 
        self.q2m = queue.Queue()  # queue to main thread
        self.q2t = queue.Queue()  # queue to a child thread
        self.sFragCI = [-1, -1]  # column indices (beginning and end) of 
          # audio data, in which average RMS amplitude went over threshold
        self.sfcis = []  # list of colmn indices of captured sound fragments
        self.sfRslts = []  # list of ('Matched', 'Unmatched' or 'N/A') 
          # whether each sound fragment matched or not, 
          # 'N/A' means no comparison was conducted.
        self.templFP = None  # folder (or file) path of template WAV file(s)
        self.templP = None  # analyzed parameters of a selected 
          # template WAV file
        self.lastTimeAmpOverThr = None  # last time 
          # when amplitude was over threshold 
        self.sfP = None  # analyzed parameters of the current 
          # sound fragment (most recent fragment captured by amplitude)
        
        self.pa = pyaudio.PyAudio()
        self.devIdx, self.devNames = self.find_device(devType='input')

        self.initSParr('both') # set up initial spectrogram arrays  

        self.isListening = False

    #-------------------------------------------------------------------

    def stop(self):
        """ Stop streaming
        """ 
        if DEBUG: print("PyListener.stop()")
        self.stream.close()
        self.rMicData = [] 
        #self.pa.terminate()
        msg = "%s, [MSG], Audio stream is closed.\n"%(get_time_stamp())
        writeFile(self.logFile, msg)

    #-------------------------------------------------------------------
   
    def find_device(self, devType='input'):
        """ Find target audio devices such as input/output, device 
        with a name with preferred name string (prefDevStr). 
        
        Args:
            devType (str): 'input' or 'output' 

        Returns: 
            devIdx (list): Found device indices (integers)
            devNames (list): Found device names (string)
        """
        if DEBUG: print("PyListener.find_device()")
        
        devIdx = []
        devNames = []
        for devStr in self.prefDevStr:
            for i in range( self.pa.get_device_count() ):     
                devInfo = self.pa.get_device_info_by_index(i)
                msg = "%s, [MSG],"%(get_time_stamp()) 
                msg += " Index:%i"%(i)
                msg += "/ device-name:%s\n"%(devInfo["name"])
                writeFile(self.logFile, msg)
                if devInfo["max%sChannels"%(devType.capitalize())] > 0:
                # if device type (input or output) matches
                    if devStr.lower() in devInfo["name"].lower():
                        msg = "%s, [MSG],"%(get_time_stamp())
                        msg += " Found an input device; device-index:%i"%(i)
                        msg += "/ device-name:%s\n"%(devInfo["name"])
                        writeFile(self.logFile, msg)
                        devIdx.append(i)
                        devNames.append(devInfo["name"])
        if devIdx == []:
            msg = "%s, [ERROR],"%(get_time_stamp())
            msg += " !! No preferred input device is found."
            msg += " Please check 'self.prefDevStr' in pyListener.py !!\n"
            writeFile(self.logFile, msg)
            print(msg)
        return devIdx, devNames

    #-------------------------------------------------------------------
    
    def open_mic_stream(self, chosenDevIdx):
        """ Open streaming with an input device index

        Args:
            chosenDevIdx (int): Device index.

        Returns:
            stream (pyaudio.Stream): Opened audio stream.
        """ 
        if DEBUG: print("PyListener.open_mic_stream()")
        stream = self.pa.open(
                                format = FORMAT,
                                channels = CHANNELS,
                                rate = RATE,
                                input = True,
                                input_device_index = self.devIdx[chosenDevIdx],
                                frames_per_buffer = INPUT_FRAMES_PER_BLOCK,
                             )
        msg = "%s, [MSG],"%(get_time_stamp())
        msg += " Stream of %i."%(self.devIdx[chosenDevIdx])
        msg += " %s is opened.\n"%(self.devNames[chosenDevIdx])
        writeFile(self.logFile, msg)
        return stream

    #-------------------------------------------------------------------
    
    def listen(self, flag='stream', wavFP=''):
        """ Read data from microphone and pre-process.
        If it's opening a wave file, read WAV file, pro-process and analyze.
        
        Args:
            flag (str): Indicates which data to process. 
              Currently, 'stream', 'wavFile' or 'templateFolder'.
            wavFP (str): Wave file path (when flag == 'wavFile') or 
              folder path (when flag == 'templateFolder'), which contains 
              WAV files for template data.

        Returns:
            data (numpy.array): Spectrogram data, which has greyscale pixel 
              values (0-255) for drawing a spectrogram.
            amp (float): RMS amplitude of data from mic.
            params (dict): Analyzed parameters of WAV data.
        """ 
        if DEBUG: print("PyListener.listen()")

        amp = None; params= None
        
        if flag == 'stream': # read from mic. stream
            try: 
                data = self.stream.read(INPUT_FRAMES_PER_BLOCK, 
                                        exception_on_overflow=False)
                self.rMicData.append(data) # store read data
            except IOError as e:
                msg = str(e)
                print(msg)
                msg = "%s, [ERROR], %s\n"%(get_time_stamp(), msg) 
                writeFile(self.logFile, msg)
                return None
            if self.frame == None: w = self.spWidth
            else: w = self.frame.pi['sp']['sz'][0]
            if len(self.rMicData) > w: self.rMicData.pop(0) # remove old data 
              # when it's out of current spectrogram width 
            amp = self.get_rms(data) # get rms amp.
            data = np.frombuffer(data, dtype=np.short) # int16
            data = self.preProcDataFromMic(data)

        elif flag == 'wavFile': # read & analyze a (non-template) WAV file
            wavData = wave.open(wavFP, 'rb')
            wp = wavData.getparams()
            wd = wavData.readframes(wp.nframes)
            wavData.close()
            wd = np.frombuffer(wd, dtype=np.dtype('i2'))
            data = self.preProcDataFromFile(wd, wp, flagInitArr=False)
            params, data = self.analyzeSpectrogramArray(data,
                                        flagTemplate=False) # analyze the sound

        elif flag in ['templateFolder', 'templateFile']: 
        # read a WAV file or WAV files in a template folder 
            chkFPath(wavFP) # check folder (file) path's existence
            if flag == 'templateFolder':
                p1 = path.join(wavFP, "*.wav")
                p2 = path.join(wavFP, "*.WAV")
                fileLists = glob(p1) + glob(p2)
            elif flag == 'templateFile':
                fileLists = [ wavFP ]
            data, params = self.formTemplate(fileLists)
            self.templP = params 
            self.tSpAD = data
        
        return data, amp, params  

    #-------------------------------------------------------------------

    def preProcDataFromMic(self, data):
        """ Pre-process wave data read from microphone.

        Args:
            data (np.array): Audio data read from microphone.

        Returns:
            data (np.array): Array contains greyscale spectrogram image. 
        """
        if DEBUG: print("PyListener.preProcDataFromMic()")
        data = data * SHORT_NORMALIZE
        data = abs(np.fft.fft(data))[:int(INPUT_FRAMES_PER_BLOCK/2)]
        maxVal = np.max(data)
        if maxVal > 1: data = data/maxVal # maximum value should be 1 
        data = (data * 255).astype(np.uint8) # make an array of 0-255 for 
          # amplitude of pixel
        data = np.flip(data) # flip to make low frequency is placed at 
          # the bottom of screen
        return data
    
    #-------------------------------------------------------------------

    def preProcDataFromFile(self, wd, wp, flagInitArr=True): 
        """ Update constants, resize array , etc on the wave file (wd) 

        Args:
            wd (np.array): WAV data read from a wave file.
            wp (namedtuple): WAV parameters such as framerate, nchannels, .. 
            flagInitArr (bool): Whether initialize spectrogram arrays. 

        Returns:
            data (np.array): Array contains greyscale spectrogram image. 
        """ 
        if DEBUG: print("PyListener.preProcDataFromFile()")
        ### update global constants
        global RATE, INPUT_FRAMES_PER_BLOCK, FREQ_RES
        RATE = wp.framerate
        INPUT_FRAMES_PER_BLOCK = int(wp.framerate * INPUT_BLOCK_TIME)
        FREQ_RES = wp.framerate / float(INPUT_FRAMES_PER_BLOCK)

        ### resize arrays
        if flagInitArr == True: self.initSParr('both')
        if self.frame != None: self.frame.onUpdateRate()

        if wp.nchannels == 2: # stereo
            wd = (wd[1::2] + wd[::2]) / 2 # stereo data to mono data
        cols = int(round(wp.nframes/float(INPUT_FRAMES_PER_BLOCK))) # number of
          # columns for array
        data = np.zeros((int(INPUT_FRAMES_PER_BLOCK/2), cols), 
                        dtype=np.uint8) # final data array
        for ci in range(cols):
            off = ci * INPUT_FRAMES_PER_BLOCK 
            ad = wd[off:off+INPUT_FRAMES_PER_BLOCK]
            ad = ad * SHORT_NORMALIZE
            ad = abs(np.fft.fft(ad))[:int(INPUT_FRAMES_PER_BLOCK/2)]
            maxVal = np.max(ad)
            if maxVal > 1.0: ad = ad/maxVal # maximum value should be 1 
            ad = (ad * 255).astype(np.uint8) # make an array of 0-255 
              # for amplitude of pixel
            ad = np.flip(ad) # flip to make low frequency is placed at 
              # the bottom of screen 
            data[:,ci] = ad

        return data  
   
    #-------------------------------------------------------------------
    
    def initSParr(self, targetSP='sp'):
        """ Initialize spectrogram array(s) and its related varialbes

        Args:
            targetSP (str): Inidicates which array to intialize. 
                'sp', 'spT' or 'both'.

        Returns:
            None
        """
        if DEBUG: print("PyListener.initSParr()")

        if self.frame == None:
            spCols = spTCols = self.spWidth
            rows = int(INPUT_FRAMES_PER_BLOCK/2)
        else:
            pi = self.frame.pi
            spCols = pi["sp"]["sz"][0]
            if hasattr(pi, 'spT'):
                spTCols = pi["spT"]["sz"][0]
            else:
                spTCols = spCols
            rows = int(INPUT_FRAMES_PER_BLOCK/2)

        if targetSP in ['sp', 'both']:
            self.sFragCI = [-1, -1]
            self.lastTimeAmpOverThr = None
            # numpy array for spectrogram
            self.spAD = np.zeros((rows, spCols), dtype=np.uint8)
        if targetSP in ['spT', 'both']:
            self.templP = None
            # numpy array for spectrogram of template WAV 
            self.tSpAD= np.zeros((rows, spTCols), dtype=np.uint8)

    #-------------------------------------------------------------------
  
    def startContMicListening(self, chosenDevIdx):
        """ Start a thread for continuous listening via microphone.

        Args:
            chosenDevIdx (int): Audio device index to open a stream.

        Returns:
            None
        """
        if DEBUG: print("PyListener.startContMicListening()")

        if not isinstance(self.spAD, np.ndarray): return
        self.isListening = True
        self.initSParr('sp')
        self.th = Thread(target=self.contMicListening, 
                         args=(self.spAD, self.q2m, self.q2t, chosenDevIdx))
        self.th.start() # start the thread 

    #-------------------------------------------------------------------
    
    def contMicListening(self, spAD, q2m, q2t, chosenDevIdx):
        """ Function for a thread for continuous listening to the microphone
        update data in a column of spAD (spectrogram data in numpy array)
        It keeps sending data via queue, 
        spAD, list of RMS amplitudes (amps), current column index of spAD (cci)

        Args:
            spAD (np.array): Spectrogram array.
            q2m (Queue): Queue to send message back.
            q2t (Queue): Queue to get sent message to this thread.
            chosenDevIdx (int): Audio device index to open.

        Returns:
            None
        """
        if DEBUG: print("PyListener.contMicListening()")
        cci = 0 # current column index for putting a audio-data column
        amps = [] # list of RMS amplitudes of recent audio data
        self.stream = self.open_mic_stream(chosenDevIdx)
        while True:
            rData = receiveDataFromQueue(q2t, self.logFile)
            if rData != None:
                if rData[0] == 'msg' and rData[1] == 'quit': break
            
            ad, amp, __ = self.listen('stream') # Listen to the mic  

            amps.append(amp)
            if len(amps) > self.ampRecLen: amps.pop(0) 

            if cci < spAD.shape[1]:
                spAD[:,cci] = ad 
                cci += 1
            else:
                ### remove 1st column and append the column of 
                ### new audio-data at the end of array
                _tmp = np.copy(spAD[:,1:])
                spAD = np.zeros(spAD.shape, dtype=np.uint8)
                spAD[:,:-1] = _tmp
                spAD[:,-1] = ad 

            if isinstance(ad, np.ndarray):
                q2m.put(('aData', (spAD, amps, cci)), True, None)
        self.stop() 

    #-------------------------------------------------------------------
    
    def contProcMicAudioData(self, q2t):
        """ This function is for when 
        there's no GUI frame to continuously process microphone data.

        Args:
            q2t (Queue): Queue to get sent message to this thread.

        Returns:
            None
        """ 
        if DEBUG: print("PyListener.contProcMicAudioData()")
        while True:
            rData = receiveDataFromQueue(q2t, self.logFile)
            if rData != None:
                if rData[0] == 'msg' and rData[1] == 'quit': break

            # process recent audio data from mic.
            sfFlag, analyzedP, sfD = self.procMicAudioData()

            if sfFlag == 'started': print("Sound fragment started.")
            elif sfFlag == 'stopped': print ("Sound fragment stopped.")
            if analyzedP != None:
            # there are analyzed parameters of sound fragment
                rsltTxt = self.logSFParms(analyzedP) # log parameters of sound
                tParams2c = {}
                for param in self.compParamList:
                    tParams2c[param+'_min'] = self.templP[param+"_min"]
                    tParams2c[param+'_max'] = self.templP[param+"_max"]
                # compare sound fragment parmaeters with template 
                rslt, _txt = self.compareParamsOfSF2T(analyzedP, tParams2c) 
                if rslt == True: # matched
                    rsltTxt += "%s\n"%(_txt) 
                    fp = self.writeWAVfile(sfD) # save the captured sound 
                      # to a wave file
                    rsltTxt += "WAV file, %s, is saved."%(fp)
                else: # didn't match
                    rsltTxt += "%s\n"%(_txt) 
                print(rsltTxt)

    #-------------------------------------------------------------------
    
    def procMicAudioData(self, isWavFile=False,
                         isLastCall=False, spAD=None, amps=None, 
                         cci=None, flagAnalyze=True):
        """ Receive mic. audio data from running thread (contMicListening), 
        and process it. This function is called by a function 
        'frame.updateSpectrogram', which runs periodically using wx.Timer.

        When a long WAV file was loaded, this function is directly called,
        without using Queue. In this case, 'data', 'amps' and 'ci' arguments
        are given.

        Args:
            isWavFile (bool): When this is True, data, amps and cci arguments 
              should be directly given.
            isLastCall(bool): When it's processing WAV file, this notifies
              that it's the end of the file.
            spAD (numpy.array, optional): Audio spectrogram data.
            amps (list, optional): List of RMS amplitudes.
            cci (int, optional): Current column index.
            flagAnalyze (bool): Whether analyze audio data or not. 

        Returns:
            sfFlag (bool): Whether sound fragment captureing started or stopped
            params (dict): Parameters of the cpatured sound fragment.
            sfD (list): Each item is string read from mic. stream. Length of 
                each item is INPUT_FRAMES_PER_BLOCK. 
        """ 
        if DEBUG: print("PyListener.procMicAudioData()")
        rData = None
        sfci = self.sFragCI
        sfFlag = ""
        params = None
        sfD = None

        if isWavFile == False:
        # Mic. data
            ### get the most recent data
            missing_msg_cnt = -1  
            while self.q2m.empty() == False:
                rData = receiveDataFromQueue(self.q2m, self.logFile)
                missing_msg_cnt += 1 # count how many queued messages 
                  # were missed 
            if rData != None and rData[0] == 'aData':
                self.spAD = rData[1][0] # spectrogram data from mic
                amps = rData[1][1] 
                cci = rData[1][2] # current column index
                  # (in which the last audio stream data was stored)
                if cci >= self.spAD.shape[1]: # spectrogram is moving
                    ### move column indcies of spectrogram
                    num = 1 + missing_msg_cnt
                    if sfci[0] > -1: sfci[0] -= num 
                    else: sfci = [-1, -1]
                    if sfci[1] > -1: sfci[1] -= num
                    for i in range(len(self.sfcis)):
                        if self.sfcis[i][0] > -1: self.sfcis[i][0] -= num
                        else:
                            self.sfcis[i] = None
                            self.sfRslts[i] = None
                        if self.sfcis[i] != None: self.sfcis[i][1] -= num
                    while None in self.sfcis: self.sfcis.remove(None)
                    while None in self.sfRslts: self.sfRslts.remove(None)
        else:
        # processing WAV file
            missing_msg_cnt = 0
            self.spAD = spAD 

        if flagAnalyze == False: return # return if no analysis is requested.

        if (rData != None and rData[0] == 'aData') or isWavFile == True:
            if isWavFile and isLastCall: amps = [0]
            if amps != [] and np.average(amps) > self.ampThr:
            # average of RMS amplitude of recent audio data is over threshold
                if self.lastTimeAmpOverThr == None:
                    sfFlag = 'started' 
                    sfci = [ max(0, cci-self.ampRecLen), -1 ] # store the 
                      # beginning index of data
                if isWavFile: self.lastTimeAmpOverThr = cci * INPUT_BLOCK_TIME
                else: self.lastTimeAmpOverThr = time()
                
            else:
            # RMS amp. is under threshold
                isEndOfSF = False
                if self.lastTimeAmpOverThr != None \
                  and time()-self.lastTimeAmpOverThr > self.maxDurLowerThr:
                # sound fragment started and 
                # amplitude was below threshold for 
                # long enough time (> self.maxDurLowerThr)
                    isEndOfSF = True
                
                if isWavFile and self.lastTimeAmpOverThr != None:
                # processing WAV file and sound fragment already started.
                    _time = cci * INPUT_BLOCK_TIME 
                    if _time-self.lastTimeAmpOverThr > self.maxDurLowerThr \
                      or isLastCall:
                    # amplitude was below threshold for long enough
                    #  or this is end of WAV file.
                        isEndOfSF = True

                if isEndOfSF: 
                    self.lastTimeAmpOverThr = None 
                    sfFlag = 'stopped'
                    sfci[1] = cci-1 # record the last column index
                    if (sfci[1]-sfci[0]) * INPUT_BLOCK_TIME >= self.minDur4SF:
                    # reached the minimum duration
                        _d = self.spAD[:,sfci[0]:sfci[1]] # sound fragment data
                          # to analyze
                        params, _d = self.analyzeSpectrogramArray(_d,
                                        flagTemplate=False) # analyze the sound
                        self.sfP = params 
                        self.spAD[:,sfci[0]:sfci[1]] = _d
                        sfD = self.rMicData[sfci[0]:sfci[1]] # get raw data
                              # (from mic.) of the analyzed sound fragment 
                        self.sfcis.append( copy(sfci) ) # store column index
                        if self.templFP == None: self.sfRslts.append('N/A')
                    else: # didn't reach minimum duration
                        sfci = [-1, -1]
           
            self.sFragCI = sfci
        return sfFlag, params, sfD

    #-------------------------------------------------------------------
    
    def endContMicListening(self):
        """ Finish the thread for continuous listening via mic.

        Args: None

        Returns: None
        """
        if DEBUG: print("PyListener.endContMicListening()")
        ### end the thread
        self.q2t.put(('msg', 'quit'), True, None) 
        self.th.join()
        self.th = None

        self.isListening = False
        self.sFragCI = [-1, -1]
        self.sfcis = []
        self.sfRslts = []

    #-------------------------------------------------------------------

    def compareWAV2Template(self, wavFP):
        """ Read a WAV file (as if it's Mic. stream) to compare its audio 
          data contents to template WAV data.

        Args:
            wavFP (str): File path of a WAV file. 

        Returns:
            None
        """
        wavData = wave.open(wavFP, 'rb') # read WAV file
        wp = wavData.getparams() # get wave parameters 
        
        global RATE, INPUT_FRAMES_PER_BLOCK
        RATE = wp.framerate
        INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

        cols = int(round(wp.nframes/float(INPUT_FRAMES_PER_BLOCK))) # number of
          # columns for array
        amps = []
        spAD = np.zeros((int(INPUT_FRAMES_PER_BLOCK/2), cols), 
                        dtype=np.uint8) # spectrogram data array
        isLastCall = False 
        savWI = 1 # index number for WAV file to save

        ### process WAV audio data as if it's a streaming data from Mic.
        for cci in range(cols):
            wd = wavData.readframes(INPUT_FRAMES_PER_BLOCK)
            self.rMicData.append(wd) # store read WAV data
            amp = self.get_rms(wd) # get rms amp.
            amps.append(amp)
            if len(amps) > self.ampRecLen: amps.pop(0)
            wd = np.frombuffer(wd, dtype=np.short) # int16
            ad = self.preProcDataFromMic(wd)
            spAD[:,cci] = ad
            if cci == cols-1: isLastCall = True
            sfFlag, analyzedP, sfD = self.procMicAudioData(True, isLastCall,
                                                           spAD, amps,
                                                           cci)
            if analyzedP != None:
            # analyzed parameters are available
                rsltTxt = self.logSFParms(analyzedP) # log paraemters 
                  # of sound fragment
                ### prepare parameters to prepare
                tParams2c = {}
                for param in self.compParamList:
                    tParams2c[param+'_min'] = self.templP[param+"_min"]
                    tParams2c[param+'_max'] = self.templP[param+"_max"]
                # compare sound fragment parmaeters with template 
                rslt, _txt = self.compareParamsOfSF2T(analyzedP, tParams2c) 
                if rslt == True:
                # matched
                    rsltTxt += "%s"%(_txt) 
                    fp = "recordings/rec_%s_%03i.wav"%(get_time_stamp(), savWI)
                    savWI += 1
                    self.writeWAVfile(sfD, fp) # save the captured sound 
                      # to a wave file
                    rsltTxt += "WAV file, %s, is saved.\n\n"%(fp)
                else:
                # didn't match
                    rsltTxt += "%s"%(_txt) 
                print(rsltTxt)
        wavData.close()
    
    #-------------------------------------------------------------------

    def get_rms(self, data):
        """ Calculates Root Mean Square amplitude.

        Args:
            data (string): Read from mic. stream.

        Returns:
            (float): Root mean square amplitude.
        """ 
        if DEBUG: print("PyListener.get_rms()")
        ### get one short out for each two chars in the string.
        count = len(data)/2
        frmt = "%dh"%(count)
        shorts = struct.unpack( frmt , data )

        # iterate over the block.
        sum_squares = 0.0
        for sample in shorts:
            # sample is a signed short in +/- 32768. 
            # normalize it to 1.0
            n = sample * SHORT_NORMALIZE
            sum_squares += n*n

        return np.sqrt( sum_squares / count )

    #-------------------------------------------------------------------
    
    def autoContrast(self, data, adjVal=20, flagTemplate=False): 
        """ Apply auto-contrast with a threshold, using threshold_li 
        (Li’s iterative Minimum Cross Entropy method) on spectrogram image.
        
        Args:
            data (numpy.array): Data to apply auto-contrast
            adjVal (int): How much increase/decrease data with the threshold

        Returns:
            data (numpy.array): Data after applying auto-contrast
        """
        if DEBUG: print("PyListener.autoContrast()")
        
        if np.sum(data) == 0: return data

        ### find threshold for auto-contrasting 
        if flagTemplate == True: tol = self.acThrTol_templ
        else: tol = self.acThrTol_nt
        acThr = filters.threshold_li(data, tolerance=tol) # detect threshold
        data = data.astype(np.float32)
        data[data<=acThr] -= adjVal 
        data[data>acThr] += adjVal 
        data[data<0] = 0 # cut off too low values
        maxVal = np.max(data)
        if maxVal > 255: data *= (255.0/maxVal)
        data = data.astype(np.uint8)
        return data
    
    #-------------------------------------------------------------------

    def analyzeSpectrogramArray(self, inputData, flagTemplate=False):
        """ Extract parameters from spectrogram data. 

        Args:
            inputData (numpy.array): Spectrogram data to analyze.
            flagTemplate (bool): Whether this is WAV data for template.

        Returns:
            params (dict): Analyzed parameters. 
            data (numpy.array): Spectrogram data after some processing 
                such as cutoff frequency, auto-contrast, etc.. 
        """
        if DEBUG: print("PyListener.analyzeSpectrogramArray()")
       
        data = np.copy(inputData)
        params = {} # result dictionary to return
        for key in self.pKeys: params[key] = -1 # initial value
        
        if np.sum(data) == 0: return params, data
        
        ### cut off data in range of frequencies, self.comp_freq_range 
        cutI1 = data.shape[0]-int(self.comp_freq_range[1]/FREQ_RES)
        cutI2 = data.shape[0]-int(self.comp_freq_range[0]/FREQ_RES)
        data[:cutI1,:] = 0 # delete high frequency range
        data[cutI2:,:] = 0 # delete low frequency range

        # auto contrasting
        data = self.autoContrast(data, 20, flagTemplate=flagTemplate) 

        ### processing each column of data
        nonZeroPts = []
        nonZeroLowestFreqRowList = []
        nonZeroHighestFreqRowList = []
        cms = [] # list of center-of-mass in each column
        for ci in range(data.shape[1]):
            a = data[:,ci]
            _nz = np.nonzero(a)[0]
            nonZeroPts.append(len(_nz))
            if len(_nz) > 0:
                nonZeroLowestFreqRowList.append(np.max(_nz))
                nonZeroHighestFreqRowList.append(np.min(_nz))
            cm = center_of_mass(a)[0]
            if np.isnan(cm) == True: cms.append(-1)
            else: cms.append(int(cm))
        ### change -1 values from CenterOfMass list to its neighbor value
        for i in range(len(cms)):
            if cms[i] == -1:
                if i == len(cms)-1: cms[i] = cms[i-1]
                else:
                    j = 1 
                    while j < len(cms)-1:
                        if cms[j] != -1:
                            cms[i] = cms[j]
                            break
                        j += 1
        
        ##### begin: calculating and storing analyzed params. -----
        ### calculate duration
        params["duration"] = INPUT_BLOCK_TIME * data.shape[1] 
        ### summed amplitude ratio
        params["summedAmp"] = np.sum(data.astype(np.int32))
        if flagTemplate == True: # loading a template WAV
            params["summedAmpRatio"] = 1.0
        else:
            if self.templP != None: # there's a template file params.
                _sumD = np.sum(data.astype(np.int32))
                params["summedAmpRatio"] = _sumD / self.templP["summedAmp"]
        ### store center-of-mass in each column
        params["cmInColList"] = cms  
        ### center-of-mass in column & row, 
        ### and in terms of relative position (0.0-1.0) 
        row, col = center_of_mass(data)
        params["centerOfMassX"] = int(col)
        params["centerOfMassY"] = int(row)
        params["cmxN"] = params["centerOfMassX"]/data.shape[1]
        params["cmyN"] = 1.0-params["centerOfMassY"]/data.shape[0]
        ### calculate permutation entropy value
        #params["permEnt"] = ent.permutation_entropy(params["cmInColList"], 
        #                                            order=5, 
        #                                            normalize=True)
        ### average number of non-zero data points in columns
        if nonZeroPts == []: params["avgNumDataInCol"] = -1
        else: params["avgNumDataInCol"] = np.average(nonZeroPts)
        ### lowest and highest non-zero row and its frequency
        if nonZeroLowestFreqRowList == []:
            params["lowFreqRow"] = -1
            params["lowFreq"] = -1
        else:
            params["lowFreqRow"] = int(np.average(nonZeroLowestFreqRowList))
            _t = data.shape[0] - params["lowFreqRow"]
            params["lowFreq"] = _t * FREQ_RES / 1000
        if nonZeroHighestFreqRowList == []:
            params["highFreqRow"] = -1
            params["highFreq"] = -1
        else:
            params["highFreqRow"] = int(np.average(nonZeroHighestFreqRowList))
            _t = data.shape[0] - params["highFreqRow"]
            params["highFreq"] = _t * FREQ_RES / 1000
        ### distance between lowFreqRow and highFreqRow
        params["distLowRow2HighRow"] = params["lowFreqRow"] - params["highFreqRow"]
        ### calculates parameters about relation between 
        ### the current sound to the template sound 
        if self.templP != None: # there's a template file params. 
            if flagTemplate == False: # this is not a template file loading
                r = -1
                _d = data.astype(np.int32) # spectrogram data of 
                  # the currently received sound 
                _t = self.tSpAD.astype(np.int32) # spectrogram data of 
                  # template sound
                ### calculates correlation to auto-correlation ratio
                autocorr = correlate(_t, _t) # auto-correlation of template
                corr = correlate(_d, _t) # correlation between two sounds 
                acm = np.max(autocorr) # max. overlapping of auto-correlation
                cm = np.max(corr) # max. overlapping value of correlation 
                r = float(cm) / acm
                if r > 1.0: r = 1.0-(r-1.0)
                params["corr2auto"] = r
        ##### end: calculating and storing analyzed params. -----
        
        return params, data

    #-------------------------------------------------------------------
  
    def formTemplate(self, fileLists):
        """ Process list of wave files to form template WAV data

        Args:
            fileLists (list): List of wave files to process

        Returns:
            data (numpy.array): Spectrogram data after analysis.
            tParams (dict): Parameters of template WAV data.
        """ 
        if DEBUG: print("PyListener.formTemplate()")
        for i in range(len(fileLists)): # go through all files
            fp = fileLists[i]
            ### read wave file
            wavData = wave.open(fp, 'rb')
            params = wavData.getparams()
            wd = wavData.readframes(params.nframes)
            wavData.close()
            wd = np.frombuffer(wd, dtype=np.dtype('i2'))
            if i == 0:
                ### the 1st file, initilization
                initFR = params.framerate
                data = np.zeros((1,1), dtype=np.uint16) # final 'data'
                d = self.preProcDataFromFile(wd, 
                                         params, 
                                         flagInitArr=True) # currnet data, 'd'
                tParams = {} # result params dictionary 
                for key in self.pKeys: tParams[key] = [] # temporarily make it 
                  # as a list to append data from all WAV files 
            else:
                ### validity checking
                if initFR != params.framerate:
                    msg =  "File '%s' was not loaded"%(fp)
                    msg += " due to its different framerate"
                    msg += " %i from the framerate"%(params.framerate)
                    msg += " %i of the first file.\n"%(initFR)
                    print(msg)
                    writeFile(self.logFile,
                              "%s, [ERROR] %s"%(get_time_stamp(), msg))
                    continue
                d = self.preProcDataFromFile(wd, 
                                         params, 
                                         flagInitArr=False) # current data, 'd'
            # analyze the current data 
            params, d = self.analyzeSpectrogramArray(d, flagTemplate=True)
            ### (temporarily) append obtained params in tParams
            for key in self.pKeys: tParams[key].append(params[key])
            ### keep the spectrogram array size same.
            if data.shape[1] != d.shape[1]: # length of current data, 'd',
              # is different from 'data'
                if data.shape[1] < d.shape[1]:
                    od = np.copy(data)
                    data = np.zeros(d.shape, dtype=np.uint16)
                    data[:od.shape[0], :od.shape[1]] = od 
                else:
                    _d = np.copy(d)
                    d = np.zeros(data.shape, dtype=np.uint8)
                    d[:_d.shape[0], :_d.shape[1]] = _d
            # store average 'data'; average pixel value for final spectrogram
            data = np.array( (data+d)/2.0, dtype=np.uint16 ) 
        data = self.autoContrast(data, adjVal=40, flagTemplate=True)
        
        ##### beginning of getting parameter's 
        ##### min, max and average values. -----
        keys = list(self.pKeys)
        ### - process list data items (currently only cmInColList) 
        ### first (for other items depending on list data)
        listDataKeys = []
        initM = self.cplInitMargin
        for key in keys:
            if type(tParams[key][0]) == list:
            # data type is list
                minKey = key + "_min"
                maxKey = key + "_max"
                ### get element-wise average values of lists
                tpLen = len(tParams[key])
                tmp = []
                for i in range(tpLen): tmp.append(len(tParams[key][i]))
                dLen = int(np.average(tmp)) # final length is average length 
                  # of items of tParams[key]
                tmpArr = np.zeros((tpLen, dLen))
                for i in range(tpLen):
                    # resize data of each item to the dLen
                    tmpArr[i,:] = transform.resize( 
                                                np.asarray(tParams[key][i]),
                                                output_shape=(dLen,), 
                                                preserve_range=True, 
                                                anti_aliasing=True 
                                                  )
                tmpArr = self.levelFarOffValues(tmpArr) # make center-of-mass
                  # values, which are out of standard deviation, 
                  # to average value 
                ### min. & max. values 
                tParams[minKey] = np.min(tmpArr, axis=0)
                tParams[maxKey] = np.max(tmpArr, axis=0)
                if key in initM.keys():
                    ### give margin to min. & max. values
                    tp = tParams[minKey] - initM[minKey]
                    tParams[minKey] = list(tp.astype(np.int16))
                    tp = tParams[maxKey] + initM[maxKey]
                    tParams[maxKey] = list(tp.astype(np.int16))
                # final values in the list will be average values
                tp = np.average(tmpArr, axis=0)
                tParams[key] = list(tp.astype(np.int16))
                listDataKeys.append(key)
        for key in listDataKeys: keys.remove(key)
        ### - process other items 
        for key in keys:
            minKey = key + "_min"
            maxKey = key + "_max"
            if key in self.compParamList:
                #if key == 'permEnt': # cmInColList was adjusted in 
                #  # this function. Calculate its permutation entroy here.
                #    tp = tParams["cmInColList"]
                #    tParams[key] = ent.permutation_entropy(tp,
                #                                           order=5,
                #                                           normalize=True)
                #    tParams[minKey] = tParams[key] - initM[minKey]
                #    tParams[maxKey] = tParams[key] + initM[maxKey]
                #elif key in self.indCPL: # This parameter is not relevant 
                
                if key in self.indCPL: # This parameter is not relevant 
                  # to changing WAV data. Simply, get values from indCPRange.
                    tParams[minKey] = self.indCPRange[minKey]
                    tParams[maxKey] = self.indCPRange[maxKey]
                    tParams[key] = (tParams[minKey]+tParams[maxKey]) / 2.0

                else: # other parameters in compParamList
                    tParams[minKey] = np.min(tParams[key]) - initM[minKey]
                    tParams[maxKey] = np.max(tParams[key]) + initM[maxKey]
                    tParams[key] = np.average(tParams[key]) # store 
                      # average value of all wave files

            else:
            # this is just for internal calculations, just store average value.
                tParams[key] = np.average(tParams[key])
        ##### end of getting parameter's min, max and average values. -----
        
        return data, tParams

    #-------------------------------------------------------------------
    
    def compareParamsOfSF2T(self, sParams, tParams, fName=''):
        """ Compare parameters of sound fragment to parameters of template WAV

        Args:
            sParams (dict): Parameters of sound frament, 
                captured from mic. streaming.
            tParams (dict): Parameters of template WAV data.
            fName (string): Name of the wave file, when sParams is not 
                a captured sound from mic. streaming, but a wave file.
        """ 
        if DEBUG: print("PyListener.compareParamsOfSF2T()")
        rslt = True # whether params. of two sounds match or not
        rsltTxt = "" 
        matchedKeys = []
        for key in tParams.keys():
            if key[-4:] == '_max': continue # work with _min key
            key = key[:-4] 
            sfV = sParams[key]
            rsltTxt += "/ %s"%(key)
            minV = tParams[key+'_min']
            maxV = tParams[key+'_max']
            if sfV < minV or sfV > maxV:
                rslt = False
                rsltTxt += " [NOT]"
            else:
                matchedKeys.append(key)
            rsltTxt += " (%.3f <= %.3f <= %.3f)"%(minV, sfV, maxV)
        rsltTxt = rsltTxt.lstrip("/")
        _txt = "Sound fragment"
        if fName != '': _txt += " (%s)"%(fName)
        if rslt == True:
            _str =  str(matchedKeys).strip('[]')
            _str = _str.replace("'","").replace(",","/")
            _txt += " [MATCHED] with following parameters ( %s )"%(_str)
            _txt += rsltTxt
            rsltTxt = "%s, [RESULT], %s\n\n"%(get_time_stamp(), _txt)
        else:
            _txt += " did [NOT] match/ " + rsltTxt
            rsltTxt = "%s, [RESULT], %s\n\n"%(get_time_stamp(), _txt)
        writeFile( self.logFile, rsltTxt )

        if rslt == True: self.sfRslts.append('Matched')
        else: self.sfRslts.append('Unmatched')

        return rslt, rsltTxt

    #-------------------------------------------------------------------
    
    def levelFarOffValues(self, arr, stdF=1.0):
        """ Make value out of standard deviation(s) to average value.
        
        Args:
            arr (numpy.array): Data to process.
            stdF (float): Number of SD for keeping original data.

        Returns:
            arr (numpy.array): Data after processing.
        """ 
        if DEBUG: print("PyListener.eliminateFarOffValues()")
        avg = np.average(arr)
        std = np.std(arr)
        arr[arr<(avg-std)*stdF] = avg
        arr[arr>(avg+std)*stdF] = avg
        return arr

    #-------------------------------------------------------------------
    
    def logSFParms(self, analyzedP):
        """ Records analyzed parameters of sound fragment in log file.

        Args:
            analyzedP (dict): Parameters of sound fragment.

        Returns:
            logTxt (string): Recorded string.
        """ 
        if DEBUG: print("PyListener.logSFParms()")

        ### record the captured sound fragment parameters 
        logTxt = "%s, [RESULT],"%(get_time_stamp())
        logTxt += " Captured sound fragment parameters./ "
        for param in self.compParamList:
            _txt = "%s:%.3f/ "%(param, analyzedP[param])
            logTxt += _txt
        logTxt = logTxt.rstrip('/ ') + "\n" 
        writeFile( self.logFile, logTxt )
        return logTxt 

    #-------------------------------------------------------------------

    def writeWAVfile(self, wData, fp=""):
        """ Save given WAV data to a file.

        Args:
            wData (list): Each item is string read from mic. stream. 
              Length of each item is INPUT_FRAMES_PER_BLOCK. 
            fp (str, optional): File path to save WAV file.

        Returns:
            fp (str): File path of the saved wave file. 
        """ 
        if DEBUG: print("PyListener.writeWAVfile()")

        if fp == "": fp = "recordings/rec_%s.wav"%(get_time_stamp())
        w = wave.open( fp, 'wb' )
        w.setparams((
                        CHANNELS, 
                        SAMPLE_WIDTH, 
                        RATE, 
                        len(wData)*INPUT_FRAMES_PER_BLOCK, 
                        'NONE', 
                        'NONE'
                    ))
        for block in wData: w.writeframes(block)
        w.close()
        msg = "%s, [RESULT],"%(get_time_stamp())
        msg += " Saved to WAV file, %s\n\n"%(fp)
        writeFile(self.logFile, msg)
        return fp
    
    #-------------------------------------------------------------------

#=======================================================================

if __name__ == "__main__": pass

