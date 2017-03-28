import numpy as np
import matplotlib.pyplot as pp
import matplotlib.colorbar as colorbar
import matplotlib.animation as animation
from progressbar import ProgressBar, Percentage, Bar

import ecogana.anacode.colormaps as cmaps
import ecogana.anacode.plot_util as pu

from sandbox.tile_images import calibration_axes
import ecoglib.vis.plot_modules as pm

from ecogana.froemke.code.prepare_data import load_experiment


# Specify a recording and a time window
dset = load_experiment(
    '2016-05-24', 'test_001', bandpass=(2, 50), 
    notches=(60,)
    )
# Set y-scale
y_scl = 500

# Set window
time_start = 141 - 0.5
time_stop = 152 - 0.5

tones = list( np.unique(dset.exp.tones) )
tone_colors = pp.cm.rainbow(np.linspace(0, 1, len(tones)))

t0, tf = map(lambda x: int( x * dset.Fs ), (time_start, time_stop))
window = (t0, tf)
trial_nums = np.where( (dset.pos_edge > t0) & (dset.pos_edge < tf) )[0]
t_marks = dset.pos_edge[ trial_nums  ] / dset.Fs
t_code = [ tones.index( dset.exp.tones[t] ) for t in trial_nums ]
t_colors = [tone_colors[ tc ] for tc in t_code ]

t_marks += dset.exp.stim_props.tone_onset

ts_slice = dset.data[:, window[0]:window[1]]

ii, jj = dset.chan_map.to_mat()
ii, jj = map(np.array, zip(*sorted(zip(ii, jj))))
c_num = dset.chan_map.lookup(ii, jj)

data_window = dset.data[c_num][:, slice(*window)]
win_size = data_window.shape[1]
tx = (np.arange(data_window.shape[1]) + window[0]) / dset.Fs

figsize = (16, 9) 
fig = pp.figure(figsize=figsize)
ax = pp.subplot2grid( (1, 6), (0, 0), colspan=5 )

pplot = pm.PagedTimeSeriesPlot(
    tx, data_window.T, win_size,
    axes=ax, figure=fig,
    plot_line_props=dict(
        linewidth=0.25, color='k'
        )
    )
pplot.n_yticks = len(dset.chan_map)
#pplot.ylim = map(round, pplot.ylim)
pplot.ax.set_yticklabels(
    ['(%d, %d)'%x for x in zip(ii, jj)],
    fontsize=8
    )

pplot.draw()
ax.set_ylabel('Site index (row, col)')

t = t_marks[0]
idx = int( t * dset.Fs ) - t0
t_line = ax.axvline(t, color='r', ls='--', lw=0.5)


ax = pp.subplot2grid( (1, 6), (0, 5) )
data_window = dset.data[:, slice(*window)]
data_window_nrm = data_window / dset.data.std(1)[:,None]
clim = (-3, 3)
cmap = cmaps.diverging_cm(clim[0], clim[1])

frame = dset.chan_map.embed(data_window_nrm[:, idx])
im = ax.imshow(frame, origin='upper', clim=clim, cmap=cmap)
cb = pp.colorbar(im, shrink = 0.25)
cb.set_label(r'$\sigma$')
ttl = ax.set_title('0.00 ms', fontsize=16)

fig.tight_layout(pad=0, w_pad=1)
fig.subplots_adjust(bottom=0.03)

pplot.ax.set_xticks([])
pos = pplot.ax.get_position()
right = pos.x1
bottom = 0.025
left = 0.03

pu.mark_axes_edge(pplot.ax, t_marks, (0.050,)*len(t_marks), t_colors)

cax = pplot.fig.add_axes( [right+0.02, bottom, 0.04, pos.y1 - bottom] )

# y_scl = 5 * pplot.current_spacing
calibration_axes(
    pplot.ax, y_scale=y_scl, 
    calib_unit=dset.units, calib_ax=cax, fontsize=16
    )


# 50 ms post tone
Fs = dset.Fs
tone_win = 0.05
tone_span = int(tone_win * Fs)
tone_tx = np.linspace(0, tone_win * 1e3, tone_span)
tone_tx = np.repeat(tone_tx, len(t_marks)).reshape(len(tone_tx), -1).T.ravel()

offsets = dset.pos_edge[trial_nums] + int(dset.exp.stim_props.tone_onset * Fs)
frame_idx = [ np.arange(tone_span) + off_ - t0 for off_ in offsets ]
frame_idx = np.array(frame_idx).ravel()
times = tone_tx * 1e-3 + np.repeat(t_marks, tone_span)

tone_labels = list()
for t in dset.exp.tones[trial_nums]:
    ts = '{0:.1f} kHz'.format(t/1000.0)
    tone_labels.extend( [ts] * tone_span )

def step_anim(i):
    t = times[i]
    tone_time = tone_tx[i]
    t_name = tone_labels[i]
    ttl.set_text('{0}\n{1:0.2f} ms'.format(t_name, tone_time))
    t_line.set_data( np.array([t, t]), np.array([0, 1]) )
    frame = dset.chan_map.embed(data_window_nrm[:, frame_idx[i]])
    im.set_data(frame)
    return (t_line, frame, ttl)

fps = Fs / 100.
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='ECoG raw data', artist='ecoglib')
writer = FFMpegWriter(
    fps=fps, metadata=metadata, codec='libx264'
    )
# do arguments that are quicktime compatible
extra_args = ['-pix_fmt', 'yuv420p', '-qp', '1']
# yuv420p looks a bit crappy, but upping the res helps
#dpi = 300
dpi = fig.dpi
writer.extra_args = extra_args
fname = 'raw_signal_demo_{0}_{1}_{2}'.format(dset.name, time_start, time_stop)
with writer.saving(fig, fname+'.mp4', dpi):
    pbar = ProgressBar(
        widgets=[Percentage(), Bar()], maxval=len(tone_tx)
    ).start()
    for n in xrange(len(tone_tx)):
        step_anim(n)
        writer.grab_frame()
        pbar.update(n)
