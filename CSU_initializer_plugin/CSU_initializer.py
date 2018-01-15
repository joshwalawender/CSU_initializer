"""
Skeleton example of a Ginga local plugin called 'MyLocalPlugin'

To enable it, run ginga with the command
    $ ginga --plugins=MyLocalPlugin

it will then be available from the "Operations" button.

"""

from ginga import GingaPlugin
from ginga.gw import Widgets

# import any other modules you want here--it's a python world!
import os
from datetime import datetime as dt
import numpy as np
from ginga import GingaPlugin, RGBImage, colors
from ginga.gw import Widgets
from ginga.misc import ParamSet, Bunch
from ginga.util import dp
from ginga.gw.GwHelp import FileSelection
from astropy.io import fits
from astropy.modeling import models, fitting
from scipy import ndimage
import socket

class CSU_initializer(GingaPlugin.LocalPlugin):

    def __init__(self, fv, fitsimage):
        """
        This method is called when the plugin is loaded for the  first
        time.  ``fv`` is a reference to the Ginga (reference viewer) shell
        and ``fitsimage`` is a reference to the specific ImageViewCanvas
        object associated with the channel on which the plugin is being
        invoked.
        You need to call the superclass initializer and then do any local
        initialization.
        """
        super(CSU_initializer, self).__init__(fv, fitsimage)

        # Load plugin preferences
        prefs = self.fv.get_preferences()
        self.settings = prefs.createCategory('plugin_CSU_initializer')
        self.settings.setDefaults(ibar_num=1,
                                  mbar_num=1,
                                  ebar_num=1,
                                  move_to_open=False,
                                  bar_dest=0.0,
                                  bar_pos=137.0,
                                 )
        self.settings.load(onError='silent')

        self.instrument_hosts = ['vm-mosfire', 'nuu', 'vm-mosfirebld']
        self.hostname = socket.gethostname().split('.')[0].lower()

        self.bars_analysis = None
        self.state_analysis = None
        self.bars_file = None
        self.state_file = None
        self.bars_header = None
        self.state_header = None

        self.layertag = 'bars-canvas'
        self.dc = fv.get_draw_classes()
        canvas = self.dc.DrawingCanvas()
        canvas.enable_draw(False)
        canvas.set_surface(self.fitsimage)
        self.canvas = canvas

        self.colornames = colors.get_colors()
        self.canvas_img = None
        
        self.mfilesel = FileSelection(self.fv.w.root.get_widget())
        
        ## Fit relationship between bar position and pixels
        tick = dt.now()
        pixels, physical = self.get_data()
        self.fit_transforms(pixels, physical)
        tock = dt.now()
        elapsed = (tock-tick).total_seconds()
#         print('Completed fit of transforms in {:.3f} s'.format(elapsed))

        ## Determine slit angle and bar center to center distance in pixels
        ## from the transformation and the known longslit positions
        ##   in longslit, bar 02 is at 145.472
        ##   in longslit, bar 92 is at 129.480
        physical = [ [145.472, self.bar_to_slit(2)],
                     [129.480, self.bar_to_slit(92)] ]
        pixels = self.physical_to_pixel(physical)
        dx = pixels[1][0] - pixels[0][0]
        dy = pixels[0][1] - pixels[1][1]
        self.slit_angle_pix = np.arctan(dx/dy)
#         print("Slit Angle on CCD = {:.3f} deg".format(self.slit_angle_pix * 180./np.pi))
        self.slit_height_pix = dy / (self.bar_to_slit(92) - self.bar_to_slit(2))
#         print("Slit Height on CCD = {:.3f} pix".format(self.slit_height_pix))


    def build_gui(self, container):
        """
        This method is called when the plugin is invoked.  It builds the
        GUI used by the plugin into the widget layout passed as
        ``container``.
        This method may be called many times as the plugin is opened and
        closed for modal operations.  The method may be omitted if there
        is no GUI for the plugin.

        This specific example uses the GUI widget set agnostic wrappers
        to build the GUI, but you can also just as easily use explicit
        toolkit calls here if you only want to support one widget set.
        """
        top = Widgets.VBox()
        top.set_border_width(4)

        # this is a little trick for making plugins that work either in
        # a vertical or horizontal orientation.  It returns a box container,
        # a scroll widget and an orientation ('vertical', 'horizontal')
        vbox, sw, orientation = Widgets.get_oriented_box(container)
        vbox.set_border_width(4)
        vbox.set_spacing(2)

        self.msg_font = self.fv.get_font("sansFont", 12)

        ## -----------------------------------------------------
        ## Acquire or Load Image
        ## -----------------------------------------------------
        fr = Widgets.Frame("Image the CSU Mask")
        vbox.add_widget(fr, stretch=0)

        btns1 = Widgets.HBox()
        btns1.set_spacing(1)

        btn_acq_im = Widgets.Button("Acquire Mask Image")
        btn_acq_im.add_callback('activated', lambda w: self.acq_mask_image())
        btns1.add_widget(btn_acq_im, stretch=0)
        btns1.add_widget(Widgets.Label(''), stretch=1)

        vbox.add_widget(btns1, stretch=0)


        ## -----------------------------------------------------
        ## Analyze Image
        ## -----------------------------------------------------
        fr = Widgets.Frame("Analyze CSU Mask Image")
        vbox.add_widget(fr, stretch=0)

        btns2 = Widgets.HBox()
        btns2.set_spacing(3)

        btn_analyze = Widgets.Button("Analyze Mask Image")
        btn_analyze.add_callback('activated', lambda w: self.analyze_mask_image())
        btns2.add_widget(btn_analyze, stretch=0)
        btns2.add_widget(Widgets.Label(''), stretch=1)

        btn_overlay = Widgets.Button("Overlay Analysis Results")
        btn_overlay.add_callback('activated', lambda w: self.overlay_analysis_results())
        btns2.add_widget(btn_overlay, stretch=0)
        btns2.add_widget(Widgets.Label(''), stretch=1)


        vbox.add_widget(btns2, stretch=0)

        ## -----------------------------------------------------
        ## Edit Analysis Results
        ## -----------------------------------------------------
        fr = Widgets.Frame("Edit Analysis Results")

        captions = [
            ("Set Bar Number", 'label',\
             'set_ebar_num', 'entry',),\
            ("Set Position", 'label',\
             'set_bar_pos', 'entry'),\
            ("Edit Bar #", 'label',\
             'ebar_num', 'llabel',
             'to', 'label',
             'bar_pos', 'llabel',
             "mm", 'label',\
             "Edit Bar", 'button'),
            ]

        w, b = Widgets.build_info(captions, orientation=orientation)
        self.w.update(b)

        ebar_num = int(self.settings.get('ebar_num', 1))
        b.ebar_num.set_text('{:2d}'.format(ebar_num))
        b.set_ebar_num.set_text('{:2d}'.format(ebar_num))
        b.set_ebar_num.add_callback('activated', self.set_ebar_num_cb)
        b.set_ebar_num.set_tooltip("Set bar number to move")

        bar_pos = float(self.settings.get('bar_pos', 0.0))
        b.bar_pos.set_text('{:+.1f}'.format(bar_pos))
        b.set_bar_pos.set_text('{:+.1f}'.format(bar_pos))
        b.set_bar_pos.add_callback('activated', self.set_bar_pos_cb)
        b.set_bar_pos.set_tooltip("Set distance to move bar")

        b.edit_bar.add_callback('activated', lambda w: self.edit_bar())

        fr.set_widget(w)
        vbox.add_widget(fr, stretch=0)



        ## -----------------------------------------------------
        ## Bar Overlay
        ## -----------------------------------------------------
        fr = Widgets.Frame("Bar Positions Overlay")
        vbox.add_widget(fr, stretch=0)

        btns1 = Widgets.HBox()
        btns1.set_spacing(1)

        btn_csu_bar_state = Widgets.Button("From csu_bar_state")
        btn_csu_bar_state.add_callback('activated', lambda w: self.overlaybars_from_file())
        btns1.add_widget(btn_csu_bar_state, stretch=0)
        btns1.add_widget(Widgets.Label(''), stretch=1)

        btn_fits_header = Widgets.Button("From FITS Header")
        btn_fits_header.add_callback('activated', lambda w: self.overlaybars_from_header())
        btns1.add_widget(btn_fits_header, stretch=0)
        btns1.add_widget(Widgets.Label(''), stretch=1)

        vbox.add_widget(btns1, stretch=0)

        btns2 = Widgets.HBox()
        btns2.set_spacing(1)

        btn_clear = Widgets.Button("Clear Overlays")
        btn_clear.add_callback('activated', lambda w: self.clear_canvas())
        btns2.add_widget(btn_clear, stretch=0)
        btns2.add_widget(Widgets.Label(''), stretch=1)

        vbox.add_widget(btns2, stretch=0)

        ## -----------------------------------------------------
        ## Initialize Bar
        ## -----------------------------------------------------
        fr = Widgets.Frame("Individual Bar Initialization")

        captions = [
            ("Set Bar Number", 'label',\
             'set_ibar_num', 'entry',),\
            ("Initialize Bar #", 'label',\
             'ibar_num', 'llabel',\
             "Initialize Bar", 'button',\
             "Open Before Init", 'checkbutton'),
            ]

        w, b = Widgets.build_info(captions, orientation=orientation)
        self.w.update(b)

        ibar_num = int(self.settings.get('ibar_num', 1))
        b.ibar_num.set_text('{:2d}'.format(ibar_num))
        b.set_ibar_num.set_text('{:2d}'.format(ibar_num))
        b.set_ibar_num.add_callback('activated', self.set_ibar_num_cb)
        b.set_ibar_num.set_tooltip("Set bar number to initialize")

        b.open_before_init.set_tooltip("Move bar to open position before initialization")
        open_before_init = self.settings.get('move_to_open', False)
        b.open_before_init.set_state(open_before_init)
        b.open_before_init.add_callback('activated', self.open_before_init_cb)
        b.initialize_bar.add_callback('activated', lambda w: self.initialize_bar())

        fr.set_widget(w)
        vbox.add_widget(fr, stretch=0)


        ## -----------------------------------------------------
        ## Move Bar
        ## -----------------------------------------------------
        # Frame for instructions and add the text widget with another
        # blank widget to stretch as needed to fill emp
        fr = Widgets.Frame("Individual Bar Control")

        captions = [
            ("Set Bar Number", 'label',\
             'set_mbar_num', 'entry',),\
            ("Set Destination", 'label',\
             'set_bar_dest', 'entry'),\
            ("Move Bar #", 'label',\
             'mbar_num', 'llabel',
             'to', 'label',
             'bar_dest', 'llabel',
             "mm", 'label',\
             "Move Bar", 'button'),
            ]

        w, b = Widgets.build_info(captions, orientation=orientation)
        self.w.update(b)

        mbar_num = int(self.settings.get('mbar_num', 1))
        b.mbar_num.set_text('{:2d}'.format(mbar_num))
        b.set_mbar_num.set_text('{:2d}'.format(mbar_num))
        b.set_mbar_num.add_callback('activated', self.set_mbar_num_cb)
        b.set_mbar_num.set_tooltip("Set bar number to move")

        bar_dest = float(self.settings.get('bar_dest', 0.0))
        b.bar_dest.set_text('{:+.1f}'.format(bar_dest))
        b.set_bar_dest.set_text('{:+.1f}'.format(bar_dest))
        b.set_bar_dest.add_callback('activated', self.set_bar_dest_cb)
        b.set_bar_dest.set_tooltip("Set distance to move bar")

        b.move_bar.add_callback('activated', lambda w: self.move_bar())

        fr.set_widget(w)
        vbox.add_widget(fr, stretch=0)


        ## -----------------------------------------------------
        ## Spacer
        ## -----------------------------------------------------

        # Add a spacer to stretch the rest of the way to the end of the
        # plugin space
        spacer = Widgets.Label('')
        vbox.add_widget(spacer, stretch=1)

        # scroll bars will allow lots of content to be accessed
        top.add_widget(sw, stretch=1)

        ## -----------------------------------------------------
        ## Bottom
        ## -----------------------------------------------------

        # A button box that is always visible at the bottom
        btns_close = Widgets.HBox()
        btns_close.set_spacing(3)

        # Add a close button for the convenience of the user
        btn = Widgets.Button("Close")
        btn.add_callback('activated', lambda w: self.close())
        btns_close.add_widget(btn, stretch=0)

        btns_close.add_widget(Widgets.Label(''), stretch=1)
        top.add_widget(btns_close, stretch=0)

        # Add our GUI to the container
        container.add_widget(top, stretch=1)
        # NOTE: if you are building a GUI using a specific widget toolkit
        # (e.g. Qt) GUI calls, you need to extract the widget or layout
        # from the non-toolkit specific container wrapper and call on that
        # to pack your widget, e.g.:
        #cw = container.get_widget()
        #cw.addWidget(widget, stretch=1)


    def close(self):
        """
        Example close method.  You can use this method and attach it as a
        callback to a button that you place in your GUI to close the plugin
        as a convenience to the user.
        """
        self.fv.stop_local_plugin(self.chname, str(self))
        return True

    def start(self):
        """
        This method is called just after ``build_gui()`` when the plugin
        is invoked.  This method may be called many times as the plugin is
        opened and closed for modal operations.  This method may be omitted
        in many cases.
        """
        # start ruler drawing operation
        p_canvas = self.fitsimage.get_canvas()
        try:
            obj = p_canvas.get_object_by_tag(self.layertag)

        except KeyError:
            # Add ruler layer
            p_canvas.add(self.canvas, tag=self.layertag)

        self.resume()

    def pause(self):
        """
        This method is called when the plugin loses focus.
        It should take any actions necessary to stop handling user
        interaction events that were initiated in ``start()`` or
        ``resume()``.
        This method may be called many times as the plugin is focused
        or defocused.  It may be omitted if there is no user event handling
        to disable.
        """
        pass

    def resume(self):
        """
        This method is called when the plugin gets focus.
        It should take any actions necessary to start handling user
        interaction events for the operations that it does.
        This method may be called many times as the plugin is focused or
        defocused.  The method may be omitted if there is no user event
        handling to enable.
        """
        pass

    def stop(self):
        """
        This method is called when the plugin is stopped.
        It should perform any special clean up necessary to terminate
        the operation.  The GUI will be destroyed by the plugin manager
        so there is no need for the stop method to do that.
        This method may be called many  times as the plugin is opened and
        closed for modal operations, and may be omitted if there is no
        special cleanup required when stopping.
        """
        pass

    def redo(self):
        """
        This method is called when the plugin is active and a new
        image is loaded into the associated channel.  It can optionally
        redo the current operation on the new image.  This method may be
        called many times as new images are loaded while the plugin is
        active.  This method may be omitted.
        """
        pass

    def __str__(self):
        """
        This method should be provided and should return the lower case
        name of the plugin.
        """
        return 'CSU Initializer Plugin'


    ## ------------------------------------------------------------------
    ##  Coordinate Transformation Utilities
    ## ------------------------------------------------------------------
    def slit_to_bars(self, slit):
        '''Given a slit number (1-46), return the two bar numbers associated
        with that slit.
        '''
        return (slit*2-1, slit*2)

    def bar_to_slit(self, bar):
        '''Given a bar number, retun the slit associated with that bar.
        '''
        return int((bar+1)/2)

    def pad(self, x):
        '''Pad array for affine transformation.
        '''
        return np.hstack([x, np.ones((x.shape[0], 1))])

    def unpad(self, x):
        '''Unpad array for affine transformation.
        '''
        return x[:,:-1]

    def fit_transforms(self, pixels, physical):
        '''Given a set of pixel coordinates (X, Y) and a set of physical
        coordinates (mm, slit), fit the affine transformations (forward and
        backward) to convert between the two coordinate systems.
        
        '''
        assert pixels.shape[1] == 2
        assert physical.shape[1] == 2
        assert pixels.shape[0] == physical.shape[0]

        # Pad the data with ones, so that our transformation can do translations too
        n = pixels.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:,:-1]
        X = pad(pixels)
        Y = pad(physical)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y)
        Ainv, res, rank, s = np.linalg.lstsq(Y, X)
        A[np.abs(A) < 1e-10] = 0
        Ainv[np.abs(A) < 1e-10] = 0
        self.Apixel_to_physical = A
        self.Aphysical_to_pixel = Ainv

    def pixel_to_physical(self, x):
        '''Using the affine transformation determined by `fit_transforms`,
        convert a set of pixel coordinates (X, Y) to physical coordinates (mm,
        slit).
        '''
        x = np.array(x)
        result = self.unpad(np.dot(self.pad(x), self.Apixel_to_physical))
        return result

    def physical_to_pixel(self, x):
        '''Using the affine transformation determined by `fit_transforms`,
        convert a set of physical coordinates (mm, slit) to pixel coordinates
        (X, Y).
        '''
        x = np.array(x)
        result = self.unpad(np.dot(self.pad(x), self.Aphysical_to_pixel))
        return result

    ## ------------------------------------------------------------------
    ##  Analyze Image to Determine Bar Positions
    ## ------------------------------------------------------------------
    def analyze_mask_image(self, filtersize=7):
        '''Loop over all slits in the image and using the affine transformation
        determined by `fit_transforms`, select the Y pixel range over which this
        slit should be found.  Take a median filtered version of that image and
        determine the X direction gradient (derivative).  Then collapse it in
        the Y direction to form a 1D profile.
        
        Using the `find_bar_edges` method, determine the X pixel positions of
        each bar forming the slit.
        
        Convert those X pixel position to physical coordinates using the
        `pixel_to_physical` method and then call the `compare_to_csu_bar_state`
        method to determine the bar state.
        '''
        ## Get image
        try:
            channel = self.fv.get_channel(self.chname)
            image = channel.get_current_image()
            data = image._get_data()
        except:
            print('Failed to load image data')
            return

        # median X pixels only (preserve Y structure)
        medimage = ndimage.median_filter(data, size=(1, filtersize))
        
        self.bars_analysis = {}
        self.state_analysis = {}
        for slit in range(1,47):
            b1, b2 = self.slit_to_bars(slit)
            ## Determine y pixel range
            y1 = int(np.ceil((self.physical_to_pixel(np.array([(4.0, slit+0.5)])))[0][1]))
            y2 = int(np.floor((self.physical_to_pixel(np.array([(270.4, slit-0.5)])))[0][1]))
            gradx = np.gradient(medimage[y1:y2,:], axis=1)
            horizontal_profile = np.sum(gradx, axis=0)
            x1, x2 = self.find_bar_edges(horizontal_profile)
            if x1 is None:
                self.bars_analysis[b1] = None
                self.state_analysis[b1] = 'UNKNOWN'
            else:
                mm1 = (self.pixel_to_physical(np.array([(x1, (y1+y2)/2.)])))[0][0]
                self.bars_analysis[b1] = mm1
                self.state_analysis[b1] = 'ANALYZED'
            if x2 is None:
                self.bars_analysis[b2] = None
                self.state_analysis[b2] = 'UNKNOWN'
            else:
                mm2 = (self.pixel_to_physical(np.array([(x2, (y1+y2)/2.)])))[0][0]
                self.bars_analysis[b2] = mm2
                self.state_analysis[b2] = 'ANALYZED'
        self.compare_to_csu_bar_state()


    def find_bar_edges(self, horizontal_profile):
        '''Given a 1D profile, dertermime the X position of each bar that forms
        a single slit.  The slit edges are found by fitting one positive and
        one negative gaussian function to the profile.
        '''
        fitter = fitting.LevMarLSQFitter()

        amp1_est = horizontal_profile[horizontal_profile == min(horizontal_profile)]
        mean1_est = np.argmin(horizontal_profile)
        amp2_est = horizontal_profile[horizontal_profile == max(horizontal_profile)]
        mean2_est = np.argmax(horizontal_profile)

        g_init1 = models.Gaussian1D(amplitude=amp1_est, mean=mean1_est, stddev=2.)
        g_init1.amplitude.max = 0
        g_init2 = models.Gaussian1D(amplitude=amp2_est, mean=mean2_est, stddev=2.)
        g_init2.amplitude.min = 0

        model = g_init1 + g_init2
        fit = fitter(model, range(0,horizontal_profile.shape[0]), horizontal_profile)

        # Check Validity of Fit
        if abs(fit.stddev_0.value) < 3 and abs(fit.stddev_1.value) < 3\
           and fit.amplitude_0.value < -1 and fit.amplitude_1.value > 1\
           and fit.mean_0.value > fit.mean_1.value:
            x1 = fit.mean_0.value
            x2 = fit.mean_1.value
        else:
            x1 = None
            x2 = None
        
        return (x1, x2)


    def compare_to_csu_bar_state(self, tol=0.3):
        if self.bars_analysis is None:
            return
        # Read csu_bar_state file
        if self.hostname in self.instrument_hosts:
            self.read_csu_bar_state('')
        else:
            print('Not running on permitted host.  Loading dummy file.')
            file = os.path.expanduser('~/MOSFIRE_Test_Data/csu_bar_state')
            self.read_csu_bar_state(file)
        # Set state for bars
        for b in range(1,93):
            # Set state_analysis to be same as state_file if not OK
            if self.state_file[b] != 'OK':
                self.state_analysis[b] = self.state_file[b]
            # otherwise check if analysis matches csu_bar_state position
            else:
                self.check_safe(b)


    def check_safe(self, b, tol=0.3):
        try:
            diff = self.bars_analysis[b] - self.bars_file[b]
            if abs(diff) < tol:
                self.state_analysis[b] = 'OK'
            else:
                if b % 2 == 0:
                    if diff > tol:
                        self.state_analysis[b] = 'DANGEROUS'
                    else:
                        self.state_analysis[b] = 'DISCREPANT'
                elif b % 2 == 1:
                    if diff < -tol:
                        self.state_analysis[b] = 'DANGEROUS'
                    else:
                        self.state_analysis[b] = 'DISCREPANT'
        except:
            self.state_analysis[b] = 'UNKNOWN'


    ## ------------------------------------------------------------------
    ##  Read Bar Positions and Overlay
    ## ------------------------------------------------------------------
    def read_csu_bar_state(self, filename):
        with open(filename, 'r') as FO:
            lines = FO.readlines()
        self.bars_file = {}
        self.state_file = {}
        state_trans = {0: 'OK', 1: 'SETUP', 2: 'MOVING', -3: 'ERROR'}
        for line in lines:
            barno, pos, statestr = line.strip('\n').split(',')
            self.bars_file[int(barno)] = float(pos)
            self.state_file[int(barno)] = state_trans[int(statestr)]

    def read_bars_from_header(self, header):
        self.bars_header = {}
        self.state_header = {}
        for i in range(1,93):
            self.bars_header[i] = float(header['B{:02d}POS'.format(i)])
            self.state_header[i] = 'FROM_FITS'

    def overlaybars(self, bars, state=None, alpha=0.8):
        colormap = {'OK': 'green',
                    'ERROR': 'red',
                    'DANGEROUS': 'red',
                    'DISCREPANT': 'yellow',
                    'UNKNOWN': 'orange',
                    'FROM_FITS': 'seagreen'}
        draw_height = 0.45
        for j in range(1, 47):
            b1, b2 = self.slit_to_bars(j)

            physical1 = [ [-2.0, j-draw_height],
                          [-2.0, j+draw_height],
                          [bars[b1], j+draw_height],
                          [bars[b1], j-draw_height] ]
            physical1 = np.array(physical1)
            pixels1 = self.physical_to_pixel(physical1)
            pixels1[2][0] += draw_height * self.slit_height_pix * np.sin(self.slit_angle_pix)
            pixels1[3][0] -= draw_height * self.slit_height_pix * np.sin(self.slit_angle_pix)

            physical2 = [ [270.4+4.0, j-draw_height],
                          [270.4+4.0, j+draw_height],
                          [bars[b2], j+draw_height],
                          [bars[b2], j-draw_height] ]
            physical2 = np.array(physical2)
            pixels2 = self.physical_to_pixel(physical2)
            pixels2[2][0] += draw_height * self.slit_height_pix * np.sin(self.slit_angle_pix)
            pixels2[3][0] -= draw_height * self.slit_height_pix * np.sin(self.slit_angle_pix)

            try:
                b1color = colormap[state[b1]]
            except:
                b1color = 'gray'
            try:
                b2color = colormap[state[b2]]
            except:
                b2color = 'gray'

            self.canvas.add(self.dc.Polygon(pixels1, color=b1color, alpha=alpha))
            self.canvas.add(self.dc.Polygon(pixels2, color=b2color, alpha=alpha))
            x1, y1 = self.physical_to_pixel([[7.0, j+0.3]])[0]
            self.canvas.add(self.dc.Text(x1, y1, '{:d}'.format(b1),
                                         fontsize=10, color=b1color))
            x2, y2 = self.physical_to_pixel([[270.4-0.0, j+0.3]])[0]
            self.canvas.add(self.dc.Text(x2, y2, '{:d}'.format(b2),
                                         fontsize=10, color=b2color))

    def overlay_analysis_results(self):
        if self.bars_analysis is None:
            return
        self.overlaybars(self.bars_analysis, state=self.state_analysis)

    def overlaybars_from_file(self):
        if self.hostname in self.instrument_hosts:
            self.read_csu_bar_state()
        else:
            print('Not running on permitted host.  Loading dummy file.')
            file = os.path.expanduser('~/MOSFIRE_Test_Data/csu_bar_state')
            self.read_csu_bar_state(file)
        self.overlaybars(self.bars_file, state=self.state_file)

    def overlaybars_from_header(self):
        ## Get header
        try:
            channel = self.fv.get_channel(self.chname)
            image = channel.get_current_image()
            header = image.get_header()
        except:
            print('Failed to load header from image')
        else:
            self.read_bars_from_header(header)
            self.overlaybars(self.bars_header, state=self.state_header)

    def clear_canvas(self):
        self.canvas.delete_all_objects()


    ## ------------------------------------------------------------------
    ##  Bar Control
    ## ------------------------------------------------------------------
    def move_bar_to_open(self, b):
        try:
            state = self.state_analysis[b]
        except:
            state = ''
        if state is 'UNKNOWN':
            print('Cannot move to open from unknown position.  No action taken.')
            return
        if b % 2 == 0:
            destination = 270.400
        elif b % 2 == 1:
            destination = 4.0
        print('Moving bar #{:02d} to {:+.2f} mm'.format(b, destination))
        cmd = 'csuMoveBar {:02d} {:.1f}'.format(b, destination)
        print(cmd)
        if self.hostname in self.instrument_hosts:
            pass
        else:
            print('Not running on permitted host.  No action taken.')


    def move_bar(self):
        bar = self.settings.get('mbar_num')
        destination = self.settings.get('bar_dest')
        print('Moving bar #{:02d} to {:+.2f} mm'.format(bar, destination))
        cmd = 'csuMoveBar {:02d} {:.1f}'.format(bar, destination)
        print(cmd)
        if self.hostname in self.instrument_hosts:
            pass
        else:
            print('Not running on permitted host.  No action taken.')


    def initialize_bar(self):
        bar = self.settings.get('ibar_num')
        if self.settings.get('move_to_open'):
            self.move_bar_to_open(bar)
        print('Initializing bar #{:02d}'.format(bar))
        cmd = 'm csuinitbar={:02d}'.format(bar)
        print(cmd)
        if self.hostname in self.instrument_hosts:
            pass
        else:
            print('Not running on permitted host.  No action taken.')


    ## ------------------------------------------------------------------
    ##  Button Callbacks
    ## ------------------------------------------------------------------
    def acq_mask_image(self):
        if self.hostname in self.instrument_hosts:
            pass
        else:
            print('Not running on permitted host.  No action taken.')

    def set_ebar_num_cb(self, w):
        ebar_num = int(w.get_text())
        self.settings.set(ebar_num=ebar_num)
        self.w.ebar_num.set_text('{:2d}'.format(ebar_num))

    def set_bar_pos_cb(self, w):
        bar_pos = float(w.get_text())
        self.settings.set(bar_pos=bar_pos)
        self.w.bar_pos.set_text('{:+.1f}'.format(bar_pos))

    def edit_bar(self):
        bar = self.settings.get('ebar_num')
        destination = self.settings.get('bar_pos')
        self.bars_analysis[bar] = destination
        self.state_analysis[bar] = 'DISCREPANT'
        self.clear_canvas()
        self.overlay_analysis_results()

    def set_ibar_num_cb(self, w):
        ibar_num = int(w.get_text())
        self.settings.set(ibar_num=ibar_num)
        self.w.ibar_num.set_text('{:2d}'.format(ibar_num))

    def set_mbar_num_cb(self, w):
        mbar_num = int(w.get_text())
        self.settings.set(mbar_num=mbar_num)
        self.w.mbar_num.set_text('{:2d}'.format(mbar_num))

    def set_bar_dest_cb(self, w):
        bar_dest = float(w.get_text())
        self.settings.set(bar_dest=bar_dest)
        self.w.bar_dest.set_text('{:+.1f}'.format(bar_dest))

    def open_before_init_cb(self, widget, tf):
        self.settings.set(move_to_open=tf)



    ## ------------------------------------------------------------------
    ##  Data to Fit Affine Transformation
    ## ------------------------------------------------------------------
    def get_data(self):
        pixels = np.array([ (1026.6847023205248, 31.815757489924671),
                            (1031.1293065907989, 31.815757489924671),
                            (1100.0527926274958, 76.568051304306408),
                            (1104.4723170387663, 76.568051304306408),
                            (869.79921202733158, 119.71402079180322),
                            (874.17468615739256, 119.71402079180322),
                            (790.04504261037619, 163.97941699869187),
                            (794.38269316256697, 163.97941699869187),
                            (844.76764696920873, 208.45498973235158),
                            (849.06840834451555, 208.45498973235158),
                            (918.16119587182891, 253.46863795483193),
                            (922.57167115281891, 253.46863795483193),
                            (667.1708458173706, 296.83477802171569),
                            (671.58750566149126, 296.83477802171569),
                            (1210.6743343816352, 342.85304935109269),
                            (1215.1047501727178, 342.85304935109269),
                            (1037.1504738673596, 386.56200191364559),
                            (1041.5376839155629, 386.56200191364559),
                            (1380.9733624348846, 431.75478066748974),
                            (1385.3923546613969, 431.75478066748974),
                            (1392.3137244788115, 476.40898670973735),
                            (1396.5838727543558, 476.40898670973735),
                            (701.99737614209846, 518.12290417047029),
                            (706.31972548163674, 518.12290417047029),
                            (775.43118955263321, 562.76481942553085),
                            (779.76336695630744, 562.76481942553085),
                            (695.39446696825667, 606.9386852721824),
                            (699.68592870194686, 606.9386852721824),
                            (1225.8966927438423, 652.79237015375304),
                            (1230.2681865131638, 652.79237015375304),
                            (1299.3047613957535, 697.52305237026349),
                            (1303.6542557465727, 697.52305237026349),
                            (953.60567493512144, 740.39597570556316),
                            (957.91890612112604, 740.39597570556316),
                            (1027.0080928255736, 784.70486151318767),
                            (1031.3650789520013, 784.70486151318767),
                            (1241.625753053888, 830.10892664282756),
                            (1245.9181149708163, 830.10892664282756),
                            (1266.796600696397, 874.17188807394371),
                            (1271.1082253968038, 874.17188807394371),
                            (1404.8881828516335, 919.85774261912377),
                            (1409.9449171925908, 919.85774261912377),
                            (1325.0207484270156, 963.32163630950686),
                            (1329.3681702175545, 963.32163630950686),
                            (1185.9570564396361, 1007.0164717446025),
                            (1190.2368155733498, 1007.0164717446025),
                            (1306.6628878384579, 1051.9073888851103),
                            (1310.9679069215179, 1051.9073888851103),
                            (1151.3860791138529, 1095.4860726831637),
                            (1155.7367238283309, 1095.4860726831637),
                            (1224.7162502034391, 1140.436681012593),
                            (1229.0598756552718, 1140.436681012593),
                            (904.70409145100268, 1183.267412335555),
                            (908.99297982589781, 1183.267412335555),
                            (978.00762214758913, 1227.9731804278615),
                            (982.41054057239705, 1227.9731804278615),
                            (869.65543493075677, 1271.3564678397893),
                            (873.95299108698168, 1271.3564678397893),
                            (942.99396243198464, 1316.2391922602001),
                            (947.36667894787513, 1316.2391922602001),
                            (1256.7806430753744, 1361.195495916817),
                            (1261.0847133245632, 1361.195495916817),
                            (1330.1305637595844, 1406.3795550431571),
                            (1334.3960288420271, 1406.3795550431571),
                            (1060.9423305503171, 1449.3586376395574),
                            (1065.3182032594575, 1449.3586376395574),
                            (1108.6465868246237, 1493.9756362677167),
                            (1112.9382994207679, 1493.9756362677167),
                            (662.84522896384874, 1536.9734554153649),
                            (667.12956877347722, 1536.9734554153649),
                            (712.5287834914659, 1581.2712766110319),
                            (716.80585127180609, 1581.2712766110319),
                            (956.48762939159371, 1626.1728182002655),
                            (960.9581522740466, 1626.1728182002655),
                            (723.23974640617337, 1670.0165354200499),
                            (727.67208274341931, 1670.0165354200499),
                            (1172.3594885486252, 1715.8650599984883),
                            (1176.8341929555718, 1715.8650599984883),
                            (1015.7329598422145, 1759.5446833817025),
                            (1020.1920698607528, 1759.5446833817025),
                            (935.82358262678224, 1803.5644982617907),
                            (940.3126440130676, 1803.5644982617907),
                            (989.98752991018682, 1847.9507718487364),
                            (994.40511955530712, 1847.9507718487364),
                            (1278.2218422583971, 1892.8072028048214),
                            (1282.7070969966558, 1892.8072028048214),
                            (1351.5377751257745, 1938.5923374638328),
                            (1355.9221844080257, 1938.5923374638328),
                            (1171.5812780061251, 1981.4914424153424),
                            (1176.0817255338613, 1981.4914424153424),
                            ])

        physical = np.array([ (139.917, self.bar_to_slit(92)),
                              (139.41,  self.bar_to_slit(91)),
                              (130.322, self.bar_to_slit(90)),
                              (129.815, self.bar_to_slit(89)),
                              (160.334, self.bar_to_slit(88)),
                              (159.827, self.bar_to_slit(87)),
                              (170.738, self.bar_to_slit(86)),
                              (170.231, self.bar_to_slit(85)),
                              (163.579, self.bar_to_slit(84)),
                              (163.072, self.bar_to_slit(83)),
                              (153.983, self.bar_to_slit(82)),
                              (153.476, self.bar_to_slit(81)),
                              (186.718, self.bar_to_slit(80)),
                              (186.211, self.bar_to_slit(79)),
                              (115.773, self.bar_to_slit(78)),
                              (115.266, self.bar_to_slit(77)),
                              (138.413, self.bar_to_slit(76)),
                              (137.906, self.bar_to_slit(75)),
                              (93.508,  self.bar_to_slit(74)),
                              (93.001,  self.bar_to_slit(73)),
                              (92.021,  self.bar_to_slit(72)),
                              (91.514,  self.bar_to_slit(71)),
                              (182.097, self.bar_to_slit(70)),
                              (181.59,  self.bar_to_slit(69)),
                              (172.502, self.bar_to_slit(68)),
                              (171.995, self.bar_to_slit(67)),
                              (182.905, self.bar_to_slit(66)),
                              (182.398, self.bar_to_slit(65)),
                              (113.665, self.bar_to_slit(64)),
                              (113.158, self.bar_to_slit(63)),
                              (104.069, self.bar_to_slit(62)),
                              (103.562, self.bar_to_slit(61)),
                              (149.161, self.bar_to_slit(60)),
                              (148.654, self.bar_to_slit(59)),
                              (139.566, self.bar_to_slit(58)),
                              (139.059, self.bar_to_slit(57)),
                              (111.528, self.bar_to_slit(56)),
                              (111.021, self.bar_to_slit(55)),
                              (108.22,  self.bar_to_slit(54)),
                              (107.713, self.bar_to_slit(53)),
                              (90.189,  self.bar_to_slit(52)),
                              (89.681,  self.bar_to_slit(51)),
                              (100.593, self.bar_to_slit(50)),
                              (100.086, self.bar_to_slit(49)),
                              (118.731, self.bar_to_slit(48)),
                              (118.223, self.bar_to_slit(47)),
                              (102.94,  self.bar_to_slit(46)),
                              (102.432, self.bar_to_slit(45)),
                              (123.212, self.bar_to_slit(44)),
                              (122.704, self.bar_to_slit(43)),
                              (113.615, self.bar_to_slit(42)),
                              (113.108, self.bar_to_slit(41)),
                              (155.354, self.bar_to_slit(40)),
                              (154.847, self.bar_to_slit(39)),
                              (145.759, self.bar_to_slit(38)),
                              (145.251, self.bar_to_slit(37)),
                              (159.887, self.bar_to_slit(36)),
                              (159.38,  self.bar_to_slit(35)),
                              (150.292, self.bar_to_slit(34)),
                              (149.785, self.bar_to_slit(33)),
                              (109.338, self.bar_to_slit(32)),
                              (108.83,  self.bar_to_slit(31)),
                              (99.742,  self.bar_to_slit(30)),
                              (99.235,  self.bar_to_slit(29)),
                              (134.842, self.bar_to_slit(28)),
                              (134.335, self.bar_to_slit(27)),
                              (128.616, self.bar_to_slit(26)),
                              (128.109, self.bar_to_slit(25)),
                              (186.778, self.bar_to_slit(24)),
                              (186.271, self.bar_to_slit(23)),
                              (180.272, self.bar_to_slit(22)),
                              (179.765, self.bar_to_slit(21)),
                              (148.417, self.bar_to_slit(20)),
                              (147.91,  self.bar_to_slit(19)),
                              (178.822, self.bar_to_slit(18)),
                              (178.314, self.bar_to_slit(17)),
                              (120.197, self.bar_to_slit(16)),
                              (119.689, self.bar_to_slit(15)),
                              (140.601, self.bar_to_slit(14)),
                              (140.094, self.bar_to_slit(13)),
                              (151.005, self.bar_to_slit(12)),
                              (150.498, self.bar_to_slit(11)),
                              (143.947, self.bar_to_slit(10)),
                              (143.44,  self.bar_to_slit(9)),
                              (106.313, self.bar_to_slit(8)),
                              (105.806, self.bar_to_slit(7)),
                              (96.717,  self.bar_to_slit(6)),
                              (96.21,   self.bar_to_slit(5)),
                              (120.202, self.bar_to_slit(4)),
                              (119.695, self.bar_to_slit(3)),
                              ])
        return pixels, physical
