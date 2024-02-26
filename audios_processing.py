import librosa
import soundfile
from scipy.io import wavfile
import os
import mne
from fast_align_audio import alignment

import matplotlib
matplotlib.use('Agg')  # TkAgg does not work
import matplotlib.pyplot as plt

import numpy as np
import csv
# from pyannote.pipeline.typing import PipelineOutput
# from huggingface_hub import notebook_login
from pyannote.audio import Pipeline
import time
import speech_recognition
import glob
import pandas as pd
import copy
import whisper
# from openai import OpenAI
import meegkit
from PIL import Image
from matplotlib import cm

# min_speech_segment_gap: second. Any two segments separated less than this should be grouped into a single segment
# min_speech_segment_length: second. Any utterance should be longer than this
MIN_SPEECH_SEGMENT_GAP = 1.0
MIN_SPEECH_SEGMENT_LENGTH = 0.1

need_transcribe = False
if need_transcribe:
  # https://github.com/pyannote/pyannote-audio
  vad_pipeline = Pipeline.from_pretrained(
      # "pyannote/voice-activity-detection",
      "pyannote/speaker-diarization-3.0",
      use_auth_token="add_toke_here")

  print(vad_pipeline)

  whisper_model = whisper.load_model("small.en")
  print(whisper_model)


def _get_subject_dirs(subject_MEG, local_dir = "./my_folder/"):
  # subject_MEG: e.g., "G02"

  # set up file and folder paths here
  data_dir = local_dir + "data/"
  processing_dir = local_dir + "processing/"

  meg_dir = data_dir + subject_MEG + "/meg/"
  eeg_dir = data_dir + subject_MEG + "/eeg/"
  audio_dir = data_dir + "audios/" + subject_MEG + "/"
  audio_dir_processing = processing_dir + "audio/" + subject_MEG + "/"
  os.system('mkdir -p ' + audio_dir_processing)

  return {
      "meg_dir": meg_dir,
      "eeg_dir": eeg_dir,
      "audio_dir": audio_dir,
      "audio_dir_processing": audio_dir_processing
  }


def get_data_for_subject(subject_MEG, local_dir = "./my_folder/"):
  # subject_MEG: e.g., "G02"

  tasks = ['B1', 'B2', 'B3', 'B4', 'B5']
  subject_data_file_list = []
  for task in tasks:
    task_data_files, subject_dirs = get_data_for_subject_task(subject_MEG, task, local_dir)
    subject_data_file_list.append(task_data_files)

  return subject_data_file_list, subject_dirs

def return_first_or_none(l):
  if len(l) > 0:
    return l[0]
  else:
    return None

def get_data_for_subject_task(subject_MEG, task, local_dir = "./my_folder/"):
  # subject_MEG: e.g., "G02"

  subject_dirs = _get_subject_dirs(subject_MEG, local_dir)
  # print(subject_dirs)

  fname_raw = glob.glob(subject_dirs["meg_dir"] + "*" + task + "*.con")[0]
  fname_elp = return_first_or_none(glob.glob(subject_dirs["meg_dir"] + "*.elp"))
  fname_hsp = return_first_or_none(glob.glob(subject_dirs["meg_dir"] + "*.hsp"))
  fname_mrk = return_first_or_none(glob.glob(subject_dirs["meg_dir"] + "*_final.mrk"))
  fname_eeg = glob.glob(subject_dirs["eeg_dir"] + "*" + task + "*.eeg")[0]
  fname_vhdr = return_first_or_none(glob.glob(subject_dirs["eeg_dir"] + "*" + task + "*.vhdr"))


  # .WAV audio data
  interviewer_audio = glob.glob(subject_dirs["audio_dir"] + "console_mic*" + task + ".wav")[0]
  participant_audio = glob.glob(subject_dirs["audio_dir"] + "subject_mic*" + task + ".wav")[0]

  return {"meg_file":fname_raw,
          "meg_file_elp":fname_elp,
          "meg_file_hsp":fname_hsp,
          "meg_file_mrk":fname_mrk,
          "eeg_file":fname_eeg,
          "eeg_file_vhdr":fname_vhdr,
                  "interviewer_audio_file": interviewer_audio,
                  "participant_audio_file": participant_audio}, subject_dirs


def _scale_data(data):
  return data/max(abs(data))   # scale float into [-1.0, 1.0]

def _keep_left_channel_only(input_file, output_dir):

  output_file = output_dir + os.path.basename(input_file).replace(".wav", "_left_channel.wav")

  if not os.path.isfile(output_file):
    print("Extracting left channel from file: %s ..." % input_file, flush=True)

    # Load as multi-channel data
    data, samplerate = soundfile.read(input_file)

    data = data.transpose()[0]  # left channel
    soundfile.write(output_file, _scale_data(data), samplerate)

    print("Done", flush=True)

  return output_file


def _extract_meg_audio(meg_file, output_dir):
  print("Extracting audio from MEG file: %s ..." % meg_file, flush=True)
  raw = mne.io.read_raw_kit(
    meg_file,
    stim=[166, 167],
    stim_code="channel",
    preload=True,
    allow_unknown_format=False
  )
  # get raw audio signal from ch166 and ch167 and save them to sepearate files
  meg_samplerate = int(raw.info["sfreq"])
  print("Meg audio sample rate: %s Hz..." % int(meg_samplerate), flush=True)

  output_file_interviewer = output_dir + os.path.basename(meg_file).replace('.con', '_interviewer.wav')
  soundfile.write(output_file_interviewer, _scale_data(raw._data[166]), meg_samplerate)

  output_file_participant = output_dir + os.path.basename(meg_file).replace('.con', '_participant.wav')
  soundfile.write(output_file_participant, _scale_data(raw._data[167]), meg_samplerate)

  print("Done", flush=True)
  return output_file_interviewer, output_file_participant


def load_meg_data(subject_task_data):

  # Raw extraction ch misc 23-29 = triggers
  # ch misc 007 = audio

  # mrk, elp and hsp need to be provided as a group (all or none)
  if None in [subject_task_data["meg_file_mrk"], subject_task_data["meg_file_elp"], subject_task_data["meg_file_hsp"]]:
    raw = mne.io.read_raw_kit(
        subject_task_data["meg_file"],
        stim=[*[166], *range(176, 190)],
        slope="+",
        stim_code="channel",
        stimthresh=2,  # 2 for adult (1 for child??)
        preload=True,
        allow_unknown_format=False,
        # verbose=True,
    )
  else:
    raw = mne.io.read_raw_kit(
      subject_task_data["meg_file"],
      mrk=subject_task_data["meg_file_mrk"],
      elp=subject_task_data["meg_file_elp"],
      hsp=subject_task_data["meg_file_hsp"],
      stim=[*[166], *range(176, 190)],
      slope="+",
      stim_code="channel",
      stimthresh=2,  # 2 for adult (1 for child??)
      preload=True,
      allow_unknown_format=False,
      # verbose=True,
  )

  return raw


def denoise_meg_data(raw):
  # Apply TSPCA for noise reduction
  noisy_data = raw.get_data(picks="meg").transpose()
  noisy_ref = raw.get_data(picks=[160,161,162]).transpose()
  data_after_tspca, idx = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]
  raw._data[0:160] = data_after_tspca.transpose()

  return raw

def filter_data(raw, l_freq=1, h_freq=40):
  # raw = my_preprocessing.reject_artefact(raw, 1, 40, False, '')
  raw.filter(l_freq=l_freq, h_freq=h_freq)
  return raw


def plot_audio_alignment(audio_file_ref, audio_file_delayed, offset_t, peak_corr, plot_sec=60):
  # read
  audio_ref_raw, ref_samplerate = soundfile.read(audio_file_ref)
  audio_delayed_raw, delayed_samplerate = soundfile.read(audio_file_delayed)

  fig, axs = plt.subplots(2, 1, sharex='all')
  t_ref_step = 1/ref_samplerate
  t_ref = np.arange(len(audio_ref_raw)) * t_ref_step
  t_delayed_step = 1/delayed_samplerate
  t_delayed = np.arange(len(audio_delayed_raw)) * t_delayed_step

  axs[0].plot(t_ref[t_ref<=plot_sec], audio_ref_raw[t_ref<=plot_sec])
  axs[0].set_title(audio_file_ref)
  axs[1].plot(t_delayed[t_delayed<=plot_sec] - offset_t, audio_delayed_raw[t_delayed<=plot_sec])
  axs[1].set_title(f'Aligned {audio_file_delayed} with peak_corr {peak_corr:.3f} and offset {offset_t:.3f} second.')


def _align_audio(audio_file_ref, audio_file_delayed, plot=False, plot_sec=60):
  max_offset = 30 # seconds
  print("Aligning timeline of %s against reference %s ..." % (audio_file_delayed, audio_file_ref), flush=True)
  # read
  audio_ref_raw, ref_samplerate = soundfile.read(audio_file_ref)
  audio_delayed_raw, delayed_samplerate = soundfile.read(audio_file_delayed)


  # downsample to match sample rate
  print(" Downsampling and truncating ...", flush=True)
  samplerate = min(ref_samplerate, delayed_samplerate)
  assert delayed_samplerate > ref_samplerate, "delayed_samplerate %d Hz should be higher than ref_samplerate %d Hz" % (delayed_samplerate, ref_samplerate)
  audio_delayed = librosa.resample(audio_delayed_raw, orig_sr=delayed_samplerate, target_sr=samplerate)
  audio_ref = audio_ref_raw

  # truncate to match length
  truncate_len = min(len(audio_delayed), len(audio_ref))
  audio_delayed = audio_delayed[0:truncate_len]
  audio_ref = audio_ref[0:truncate_len]
  print(" Done", flush=True)
  print(" Sample rate: %s Hz" % int(samplerate))
  print(" Audio length: %s min" % int(truncate_len/samplerate/60))

  # align with peak correlation
  print(" Finding offset ...", flush=True)
  offset_steps, peak_corr =  alignment.find_best_alignment_offset_with_corr_coef(
    reference_signal=np.float32(audio_ref),
    delayed_signal=np.float32(audio_delayed),
    min_offset_samples=-max_offset*samplerate,
    max_offset_samples=0,
    consider_both_polarities=True,
  )
  print(" Done", flush=True)

  # save results next to the delayed file as csv
  t_step = 1/samplerate
  offset_t = offset_steps * t_step
  print("Offset time for %s : %g second with peak corr %g." % (audio_file_delayed, offset_t, peak_corr))

  # visualization
  if plot:
    plot_audio_alignment(audio_file_ref, audio_file_delayed, offset_t, plot_sec=plot_sec)

  print("Done", flush=True)

  return offset_t, peak_corr

def _get_offset_file(raw_audio_file):
   return raw_audio_file.replace('.wav', '_offset.csv').replace('audios', 'audios_metadata')

def _save_offset(offset_t, peak_corr, raw_audio_file):
  # save the result next to the raw audio file
  output_file = _get_offset_file(raw_audio_file)
  with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([["offset", "peak_corr"]])
    writer.writerows([[offset_t, peak_corr]])
  return output_file

def read_offset(offset_file):
  df = pd.read_csv(offset_file)
  return df['offset'][0], df['peak_corr'][0]

def _segement_speech(audio_file, plot=False):
  # apply pretrained pipeline
  # takes about 5 mins
  print("Segmenting speech ...(about 5 mins)...", flush=True)
  diarization = vad_pipeline(audio_file, num_speakers=1)
  print("Done", flush=True)

  segments_reformatted = []
  for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker == 'SPEAKER_00':
      segments_reformatted.append([turn.start, turn.end])
  segments = segments_reformatted

  if plot:
    plot_segments(audio_file, segments, 0, 120)

  return segments

def postprocess_segments(segments_raw, min_speech_segment_gap=MIN_SPEECH_SEGMENT_GAP, min_speech_segment_length=MIN_SPEECH_SEGMENT_LENGTH):
  # min_speech_segment_gap: second. Any two segments separated less than this should be grouped into a single segment
  # min_speech_segment_length: second. Any utterance should be longer than this

  segments = copy.deepcopy(segments_raw)


  # merge segments separated by too small a gap
  i_merge_with_previous = []
  for i, seg in enumerate(segments):
    if i > 0:
      seg_prev = segments[i-1]
      if seg[0] - seg_prev[1] <= min_speech_segment_gap:
        i_merge_with_previous.append(i)

  merge_to_seg_ind_list = []
  for i, i_merge in enumerate(i_merge_with_previous):
    merge_to_seg_ind = i_merge - 1
    while merge_to_seg_ind >= 0:
      if merge_to_seg_ind in i_merge_with_previous:
        merge_to_seg_ind -= 1
      else:
        break
    merge_to_seg_ind_list.append(merge_to_seg_ind)

  if len(i_merge_with_previous) > 0:
    print(f"Merging {len(merge_to_seg_ind_list)} segments that are separated by too small a gap...", flush=True)
  for ind, i_merge in enumerate(i_merge_with_previous):
      merge_to_seg_ind = merge_to_seg_ind_list[ind]
      segments[merge_to_seg_ind] = [segments[merge_to_seg_ind][0], segments[i_merge][1]]

  for i_merge in sorted(i_merge_with_previous, reverse=True):
    del segments[i_merge]


  # remove segments with too short a duration
  i_remove_list = []
  for i, seg in enumerate(segments):
    if seg[1] - seg[0] < min_speech_segment_length:
      i_remove_list.append(i)
  if len(i_remove_list) > 0:
    print(f"Removing {len(i_remove_list)} segments that are too short...", flush=True)
  for i_remove in sorted(i_remove_list, reverse=True):
    del segments[i_remove]

  return segments

def test_postprocess_segments():

  segments_raw = [[0, 1], [2, 3], [3.1, 4], [4.1, 5], [6, 7], [7.1, 8], [9, 10], [10.1, 10.15], [10.3, 11], [12, 13], [14, 14.05], [15, 16]]

  segments = postprocess_segments(segments_raw, min_speech_segment_gap=0.3, min_speech_segment_length=0.1)
  assert segments == [[0, 1], [2, 5], [6, 8], [9, 11], [12, 13], [15, 16]]

test_postprocess_segments()


def _get_segments_file_raw(audio_file):
   return audio_file.replace('.wav', '_segments_raw.csv').replace('audios', 'audios_metadata')

def _get_segments_file(audio_file):
   return audio_file.replace('.wav', '_segments.csv').replace('audios', 'audios_metadata')

def _get_transcript_file(audio_file):
   return audio_file.replace('.wav', '_segments_with_transcript.csv').replace('audios', 'audios_metadata')

def _save_transcript_file(segments_with_transcript, output_file):
  with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([["index", "start", "end", "text"]])
    for ind, seg in enumerate(segments_with_transcript):
      writer.writerows([[ind, seg[0], seg[1], seg[2]]])
  return output_file

def postprocess_and_save_segment(segments_raw, audio_file):
  segments_processed = postprocess_segments(segments_raw)

  output_file_raw = _get_segments_file_raw(audio_file)
  with open(output_file_raw, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([["index", "start", "end"]])
    for ind, seg in enumerate(segments_raw):
      writer.writerows([[ind, seg[0], seg[1]]])

  output_file = _get_segments_file(audio_file)
  with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([["index", "start", "end"]])
    for ind, seg in enumerate(segments_processed):
      writer.writerows([[ind, seg[0], seg[1]]])


def read_segments(segments_file):
  df = pd.read_csv(segments_file)
  segments = []
  for _, row in df.iterrows():
    segments.append([row['start'], row['end']])
  return segments


def plot_segments(audio_file, segments, plot_start_sec, plot_end_sec):
    # read wav file
    wav_raw, samplerate = soundfile.read(audio_file)

    t_step = 1/samplerate
    t = np.arange(len(wav_raw)) * t_step

    fig, axs = plt.subplots(1, 1, sharex='all', figsize=(30, 15))

    wav_raw = wav_raw[int(plot_start_sec*samplerate):int(plot_end_sec*samplerate)]
    t = t[int(plot_start_sec*samplerate):int(plot_end_sec*samplerate)]

    axs.plot(t, wav_raw)

    for seg in segments:
      if plot_start_sec<= seg[0] < seg[1] <= plot_end_sec: #  and seg[1] - seg[0] > 0.1:
        axs.axvspan(seg[0], seg[1], color='green', alpha=0.2)
    axs.set_title('Detected speech segments for %s' % audio_file)

    formatter = matplotlib.ticker.FuncFormatter(lambda time_sec, data: time.strftime('%M:%S', time.gmtime(time_sec)))
    axs.xaxis.set_major_formatter(formatter)

    fig.show()


def export_audio_segments(segments, audio_file):
  data, samplerate = soundfile.read(audio_file)
  for seg_index, seg in enumerate(segments):
    data_seg = data[int(seg[0]*samplerate):int(seg[1]*samplerate)]
    output_file = audio_file.replace(".wav", f"_seg_{seg_index:0>4}.wav")
    soundfile.write(output_file, data_seg, samplerate)


def _transcribe_speech(segments, audio_file):
  segments_with_transcripts = copy.deepcopy(segments)
  for seg_index, seg in enumerate(segments_with_transcripts):
    seg_audio_file = audio_file.replace(".wav", f"_seg_{seg_index:0>4}.wav")
    transcript = whisper_model.transcribe(seg_audio_file)
    print(f"[Segment #{seg_index:0>4}] between {seg[0]:.3f} and {seg[1]:.3f}:")
    print(transcript["text"])
    segments_with_transcripts[seg_index].append(transcript["text"])
  return segments_with_transcripts


def audio_processing_pipeline(data_files, processing_dir, plot=False, force_align=False):
  # data_files = {
  #     "meg_file": './my_folder/data/G02/meg/20230412_Pilot07_B5.con',
  #     "interviewer_audio_file": './my_folder/data/G02/audio/console_mic_B5.wav',
  #     "participant_audio_file": './my_folder/data/G02/audio/subject_mic_B5.wav',
  # }
  # processing_dir: such as './my_folder/processing/audio/G02/'

  meg_file = data_files["meg_file"]
  interviewer_audio_file_raw = data_files["interviewer_audio_file"]
  participant_audio_file_raw = data_files["participant_audio_file"]

  interviewer_audio_file_meg, participant_audio_file_meg = _extract_meg_audio(meg_file, processing_dir)

  # Remove the right channel to avoid noise from the other person
  # Both interviewer_audio_raw and participant_audio_raw have left channel as the intended recording channel
  for audio_file, meg_audio_file in [(interviewer_audio_file_raw, interviewer_audio_file_meg),(participant_audio_file_raw, participant_audio_file_meg)]:

    left_channel_audio_file = _keep_left_channel_only(audio_file, processing_dir)

    if os.path.isfile(_get_offset_file(audio_file)) and not force_align:
      print("skipping offset detection (results already exist) for: %s" % audio_file, flush=True)
    else:
      audio_offset, peak_corr = _align_audio(meg_audio_file, left_channel_audio_file, plot=plot)
      audio_offset_file = _save_offset(audio_offset, peak_corr, audio_file)

    if os.path.isfile(_get_segments_file(audio_file)):
      print("skipping speech segmentation (results already exist) for: %s" % audio_file, flush=True)
      segments_file = _get_segments_file(audio_file)
      segments = read_segments(segments_file)
    else:
      segments = _segement_speech(left_channel_audio_file, plot=plot)
      postprocess_and_save_segment(segments, audio_file)

    if os.path.isfile(_get_transcript_file(audio_file)):
      print("skipping speech-to-text (results already exist) for: %s" % audio_file, flush=True)
    else:
      export_audio_segments(segments, left_channel_audio_file)
      segments_with_transcripts = _transcribe_speech(segments, left_channel_audio_file)
      transcript_file = _get_transcript_file(audio_file)
      _save_transcript_file(segments_with_transcripts, transcript_file)

def plot_meg_layout(data):
  layout = mne.channels.find_layout(data.info)
  layout.plot()
  # print(layout.pos.shape)

  fig, axs = plt.subplots(1, 1, sharex='all') #, figsize=(12, 10))
  x = layout.pos[0:160,0]
  y = layout.pos[0:160,1]

  axs.plot(x, y, "o--")


def get_all_subjects():
  # returns ['G01", 'G02", ...]
  all_subjects = []
  return [f"G{i:0>2}" for i in np.arange(1,33)]


def re_process_segments(subject, local_dir = "./my_folder/"):
  # Run this if min_speech_segment_gap or min_speech_segment_length is changed
  subject_data_file_list, subject_dirs = get_data_for_subject(subject, local_dir)
  for data_file in subject_data_file_list:
    interviewer_audio_file_raw = data_file["interviewer_audio_file"]
    segments_file = _get_segments_file_raw(interviewer_audio_file_raw)
    postprocess_and_save_segment(read_segments(segments_file), interviewer_audio_file_raw)

    participant_audio_file_raw = data_file["participant_audio_file"]
    segments_file = _get_segments_file_raw(participant_audio_file_raw)
    postprocess_and_save_segment(read_segments(segments_file), participant_audio_file_raw)



def load_processed_audio_data(subject, task, local_dir = "./my_folder/", inspect_alignment=False, inspect_segmentation=False):
  task_data_files, subject_dir = get_data_for_subject_task(subject, task, local_dir)

  processing_dir=subject_dir["audio_dir_processing"]

  meg_file = task_data_files["meg_file"]
  interviewer_audio_file_raw = task_data_files["interviewer_audio_file"]
  participant_audio_file_raw = task_data_files["participant_audio_file"]

  output = {}

  if inspect_alignment:
    interviewer_audio_file_meg, participant_audio_file_meg = _extract_meg_audio(meg_file, processing_dir)
    output["interviewer_audio_file_meg"] = interviewer_audio_file_meg
    output["participant_audio_file_meg"] = participant_audio_file_meg

  for prefix, audio_file in [("interviewer_", interviewer_audio_file_raw),("participant_", participant_audio_file_raw)]:
    left_channel_audio_file = _keep_left_channel_only(audio_file, processing_dir)
    offset_file = _get_offset_file(audio_file)
    offset_t, peak_corr = read_offset(offset_file)
    output[prefix+"audio_file"] = left_channel_audio_file
    output[prefix+"offset_t"] = offset_t
    output[prefix+"peak_corr"] = peak_corr
    segments_file = _get_segments_file(audio_file)
    segments = read_segments(segments_file)
    output[prefix+"segments"] = segments

  if inspect_alignment:
    for prefix in ["interviewer_", "participant_"]:
      plot_audio_alignment(output[prefix + "audio_file_meg"], output[prefix+"audio_file"], output[prefix+"offset_t"], output[prefix+"peak_corr"], plot_sec=60)
  if inspect_segmentation:
    for prefix in ["interviewer_", "participant_"]:
      plot_segments(output[prefix+"audio_file"], output[prefix+"segments"], plot_start_sec=0, plot_end_sec=120)
      export_audio_segments(output[prefix+"segments"], output[prefix+"audio_file"])

  return output



def find_meg_trigger(meg_data, task):
  task_to_trigger_channel_mapping = {
      'B1': 181,
      'B2': 182,
      'B3': 183,
      'B4': 184,
      'B5': 185,
  }
  trigger_ch = task_to_trigger_channel_mapping[task]
  trigger_data = meg_data._data[trigger_ch,:]
  std = np.std(trigger_data)
  mean = np.mean(trigger_data)
  trigger_onset = np.nonzero(trigger_data > mean+3*std)[0][0]
  return trigger_onset

def find_eeg_trigger(eeg_data):
  eeg_events, _ = mne.events_from_annotations(eeg_data)
  print("EEG events for trigger:")
  print(eeg_events)
  trigger_onset = 0
  for event in eeg_events:
    if event[0] > 0:
      trigger_onset = event[0]

  return trigger_onset


def load_eeg_data(data_file):
  # Read raw EEG data
  raw_eeg = mne.io.read_raw_brainvision(data_file["eeg_file_vhdr"], preload=True)

  # set channel types explicitly as these are not read in automatically
  raw_eeg.set_channel_types({'32': 'ecg', '63': 'eog'})

  # Filtering & ICA
  # raw_eeg = my_preprocessing.reject_artefact(raw_eeg, 1, 40, False, '')

  return raw_eeg

def normalize_data_for_plot(data):
  # zero mean
  color_data = data - data.mean()

  # normalize range to be within [0, 1]
  # (1) keep the mean at 0.5 (so that mean is plot as white)
  # (2) make sure at least one of min/max is at either 0 or 1 to fully utilize the colormap (max contrast)
  min = color_data.min()
  max = color_data.max()
  if abs(min) >= max:
    color_data = ((color_data / abs(min) ) + 1)/2
    assert color_data.min() == 0
  else:
    color_data = ((color_data / max ) - 1)/2 + 1
    assert color_data.max() == 1
  return color_data


ECG_EOG_channels = [31, 62]
EEG_channels = [c for c in list(range(63)) if c not in ECG_EOG_channels]
MEG_channels = list(range(160))
def plot_data_as_image(data, data_type, samplerate=10):
  plot_every_ms = int(1000/samplerate)
  if data_type == 'meg':
    color_data = data._data[MEG_channels, 0::plot_every_ms]
  elif data_type == 'eeg':
    color_data = data._data[EEG_channels, 0::plot_every_ms]
  else:
    print('unknown data type')

  color_data = normalize_data_for_plot(color_data)
  # Creates PIL image
  img = Image.fromarray(np.uint8(cm.seismic(color_data)*255))
  return img, color_data

def pad_color_data(meg_color_data, eeg_color_data):
  meg_length = len(meg_color_data[0,:])
  eeg_length = len(eeg_color_data[0,:])
  merged_length = max(meg_length, eeg_length)
  print("meg_length %s  eeg_length %s" % (meg_length, eeg_length))
  if merged_length - meg_length > 0:
    meg_color_data = np.append(
        meg_color_data,
        np.ones((len(meg_color_data[:,0]), merged_length - meg_length))*0.5,
        axis=1)
  elif merged_length - eeg_length > 0:
    eeg_color_data = np.append(
        eeg_color_data,
        np.ones((len(eeg_color_data[:,0]), merged_length - eeg_length))*0.5,
        axis=1)
  print("meg_color_data shape: " )
  print(meg_color_data.shape)
  print("eeg_color_data shape: ")
  print(eeg_color_data.shape)

  return meg_color_data, eeg_color_data, merged_length

def generate_segment_color_data(audio_data, samplerate, length):
  plot_every_ms = int(1000/samplerate)
  time = np.arange(length) * plot_every_ms
  segment_line_with = 5
  segment_color_data_list = []
  for segments, offset_t, color in [(audio_data["participant_segments"], audio_data["participant_offset_t"], 0.7),
   (audio_data["interviewer_segments"], audio_data["interviewer_offset_t"], 0.3)]:
    segment_data = np.ones(len(time)) * 0.5
    for seg in segments:
      t_start = seg[0] - offset_t
      t_end = seg[1] - offset_t
      segment_data[int(t_start*samplerate):int(t_end*samplerate)] = color
    segment_color_data_list.append(np.tile(segment_data, (segment_line_with, 1)))

  segment_color_data = np.append(segment_color_data_list[0], segment_color_data_list[1], axis=0)

  return segment_color_data

def extract_meg_eeg_colordata(task,meg_data,eeg_data,samplerate):
  plot_every_ms = int(1000/samplerate)

  meg_trigger_onset = find_meg_trigger(meg_data, task)
  eeg_trigger_onset = find_eeg_trigger(eeg_data)
  assert eeg_trigger_onset > 0, "eeg_trigger_onset %s should not be zero"

  print("meg_trigger_onset: %s, eeg_trigger_onset: %s" % (meg_trigger_onset, eeg_trigger_onset))

  meg_color_data = meg_data._data[0:160, 0::plot_every_ms]
  meg_color_data = normalize_data_for_plot(meg_color_data)
  if  eeg_trigger_onset > meg_trigger_onset:
    # trim eeg data (we don't pad meg data because audio segments are aligned with meg timeline)
    eeg_color_data = eeg_data._data[0:63, int(eeg_trigger_onset-meg_trigger_onset)::plot_every_ms]
    eeg_color_data = normalize_data_for_plot(eeg_color_data)
  else:
    # pad eeg data
    eeg_color_data = eeg_data._data[0:63, 0::plot_every_ms]
    eeg_color_data = normalize_data_for_plot(eeg_color_data)
    eeg_pad_length = int(np.round((meg_trigger_onset - eeg_trigger_onset)/plot_every_ms))
    print("eeg_pad_length: %s" % eeg_pad_length)
    eeg_color_data = np.append(
        np.ones((63, eeg_pad_length))*0.5,
        eeg_color_data,
        axis=1)

  print("meg_color_data.shape:")
  print(meg_color_data.shape)

  print("eeg_color_data.shape:")
  print(eeg_color_data.shape)

  meg_color_data, eeg_color_data, length = pad_color_data(
      meg_color_data, eeg_color_data)

  return meg_color_data, eeg_color_data, length



def plot_data_with_speech_segmentation_as_image(task,meg_data,eeg_data,audio_data,
                                                samplerate=100):

  meg_color_data, eeg_color_data, length = extract_meg_eeg_colordata(task,meg_data,eeg_data,
                                                samplerate)

  # plot meg data + segments first
  segment_color_data = generate_segment_color_data(audio_data, samplerate, length)

  color_data = np.append(meg_color_data, segment_color_data, axis=0)

  # plot time ticks
  time_second_tick = np.ones(length) * 0.5
  time_second_tick[::samplerate] = 0.7
  time_tick_color_data = np.tile(time_second_tick, (4, 1))
  color_data = np.append(color_data, time_tick_color_data, axis=0)

  # plot eeg
  color_data = np.append(color_data, eeg_color_data, axis=0)

  # Creates PIL image
  img = Image.fromarray(np.uint8(cm.seismic(color_data)*255))
  return img


def plot_all_data_together(subject, task, local_dir="./my_folder/"):

  data_file, subject_dirs = get_data_for_subject_task(subject, task, local_dir)
  meg_data = load_meg_data(data_file)
  meg_data = denoise_meg_data(meg_data)
  meg_data = filter_data(meg_data)
  audio_data = load_processed_audio_data(subject, task)
  eeg_data = load_eeg_data(data_file)
  eeg_data = filter_data(eeg_data)

  img = plot_data_with_speech_segmentation_as_image(task, meg_data, eeg_data, audio_data)
  return img

def get_all_tasks():
  return ['B1', 'B2','B3','B4','B5']



def get_subject_task_audio_metadata(subject, task):
    participant_file = '/content/my_folder/data/audios_metadata/%s/subject_mic_%s_segments_with_transcript.csv' % (subject, task)
    interviewer_file = '/content/my_folder/data/audios_metadata/%s/console_mic_%s_segments_with_transcript.csv' % (subject, task)

    participant_segments = read_transcript_file(participant_file)
    interviewer_segments = read_transcript_file(interviewer_file)

    participant_offset, _ = read_offset('/content/my_folder/data/audios_metadata/%s/subject_mic_%s_offset.csv' % (subject, task))
    interviewer_offset, _ = read_offset('/content/my_folder/data/audios_metadata/%s/console_mic_%s_offset.csv' % (subject, task))


    participant_segments = apply_segment_offset(participant_segments, participant_offset)
    interviewer_segments = apply_segment_offset(interviewer_segments, interviewer_offset)

    return participant_segments, interviewer_segments

def apply_segment_offset(segments, offset):
  return [[seg[0]-offset, seg[1]-offset, seg[2]] for seg in segments]


def combine_segments(participant_segments, interviewer_segments):
  segment = []
  p_i = 0
  i_i = 0
  p_len = len(participant_segments)
  i_len = len(interviewer_segments)
  while p_i < p_len or i_i < i_len:
    if p_i == p_len:

      segment.append(["interviewer"] + interviewer_segments[i_i])
      i_i += 1
    elif i_i == i_len:
      segment.append(["participant"] + participant_segments[p_i])
      p_i += 1

    elif participant_segments[p_i] < interviewer_segments[i_i]:
      segment.append(["participant"] + participant_segments[p_i])
      p_i += 1
    else:
      segment.append(["interviewer"]+ interviewer_segments[i_i])
      i_i += 1
  return segment


def read_transcript_file(file):
  if os.path.isfile(file):
    df = pd.read_csv(file)
    segments = []
    for _, row in df.iterrows():
      segments.append([row['start'], row['end'], row['text']])
  else:
    # read fall-back file without transcript
    df = pd.read_csv(file.replace('_with_transcript',''))
    segments = []
    for _, row in df.iterrows():
      segments.append([row['start'], row['end'], ''])  # empty string for transcript

  return segments

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

def _save_labelled_transcript_file(segments, output_file):
  with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([["index", "person", "start", "end", "text", "is_full_turn", "encloses"]])
    for ind, seg in enumerate(segments):
      writer.writerows([[ind, seg[0], seg[1], seg[2],  seg[3], seg[4], seg[5]]])
  return output_file


def plot_and_save_conversation(subject, task):
  participant_segments, interviewer_segments = get_subject_task_audio_metadata(subject, task)
  segments = combine_segments(participant_segments, interviewer_segments)

  segments = label_turn_taking(segments)
  _save_labelled_transcript_file(segments, '/content/my_folder/data/audios_metadata_labelled/%s_%s.csv' % (subject, task))


  fig, axe = plt.subplots(figsize=(72,24))
  axe.invert_yaxis()
  #Spacing between each line
  intervals = float(5)
  loc = plticker.MultipleLocator(base=intervals)
  axe.xaxis.set_major_locator(loc)

  axe.grid(axis="x")

  for index, seg in enumerate(segments):
      y = index
      if seg[0] == 'interviewer':
        c = 'tan'
      else:
        c = 'pink'
      x_start = seg[1]
      axe.broken_barh([(x_start, seg[2] - seg[1])],
                      (y-0.5,0.8),
                      facecolors =(c))
      if seg[4] == True: # tagged as a full-turn in conversation
        axe.text(x_start+0.1, y+0.1, seg[3])
      else:
        # use bracket to mark a non-full-turn (e.g. back-channeling such as "yeah, hmm, ok")
        axe.text(x_start+0.1, y+0.1, "(%s)" % (seg[3]))

  plt.savefig('/content/my_folder/data/audios_metadata_plot/%s_%s.png' % (subject, task))

def label_turn_taking(segments):
  def _is_seg_enclosed_by_others(i_seg, seg, segments):
    ## seg_a = ['interviewer', 2, 4, 'adfdfff']
    ## seg_b = ['participant', 1, 6, 'oadfak asdfa asdf']
    ## We say seg_a is enclosed by seg_
    ## Also note that all segments are sorted by start_time

    for i in range(i_seg):
      if segments[i][2] > seg[2]: # segments[i][1] < seg[1] will always be true
        return True, i
    return False, None

  for i, seg in enumerate(segments):
    segments[i].extend([None, []]) # the list of segments that it encloses
    is_enclosed, enclosed_by = _is_seg_enclosed_by_others(i, seg, segments)
    if is_enclosed: ## An enclosed segment is not a "full-turn" in a two-person conversation
      segments[i][4] = False
      segments[enclosed_by][5].append(i)
    else:
      ## Otherwies tag it as a full-turn
      segments[i][4] = True

  return segments

def get_participant_turns(file):
  min_event_duration = 2 # seconds
  df = pd.read_csv(file)
  participant_turns = []
  for _, row in df.iterrows():
    if row['person'] == 'participant' \
        and (row['end']-row['start']) > min_event_duration \
        and bool(row['is_full_turn']):

      participant_turns.append([row['start'],row['end']])

  return participant_turns


def plot_top_channels(subject, bad_channels=None, denoise=False, top_n_channels_input=None):
  data_files, subject_dir = get_data_for_subject(subject)

  top_n_channels_output = {'eeg':{}, 'meg':{}}

  fig_eeg, ax = plt.subplots(5,1,figsize=(60, 40), sharex='all')
  for i, data_file in enumerate(data_files):
    eeg_data = load_eeg_data(data_file)
    if denoise:
      eeg_data = filter_data(eeg_data, l_freq=1, h_freq=40)

    bad_channels_here = []
    if bad_channels:
      task = 'B' + str(i+1)
      try:
        bad_channels_here = bad_channels['eeg'][subject][task]
        print("bad_channels for % %s: %s" %(subject, task, bad_channels_here))
      except:
        pass

    good_eeg_channels = [c for c in EEG_channels if c not in bad_channels_here]

    if top_n_channels_input is None:
      top_n_channels = get_channels_with_top_magnitude(eeg_data._data[good_eeg_channels,:])
      top_n_channels_output['eeg'][i] = top_n_channels
    else:
      top_n_channels = top_n_channels_input['eeg'][i]
      top_n_channels = [c for c in  top_n_channels if c not in bad_channels_here]
    plot_data(eeg_data, top_n_channels, ax[i])

  prefix = ""
  if bad_channels:
    prefix += "_bad_channels_removed"
  if denoise:
    prefix += "_denoised"

  fig_name = '/content/my_folder/data/plots/%s_top_eeg_channels%s.png' % (subject, prefix)
  fig_eeg.savefig(fig_name)

  print('Saving eeg figs %s...' % fig_name)



  fig_meg, ax = plt.subplots(5,1,figsize=(60, 40), sharex='all')
  for i, data_file in enumerate(data_files):
    meg_data = load_meg_data(data_file)
    if denoise:
      meg_data = denoise_meg_data(meg_data)
      meg_data = filter_data(meg_data, l_freq=1, h_freq=40)

    bad_channels_here = []
    if bad_channels:
      task = 'B' + str(i+1)
      try:
        bad_channels_here = bad_channels['meg'][subject][task]
        print("bad_channels for % %s: %s" %(subject, task, bad_channels_here))
      except:
        pass

    good_meg_channels = [c for c in MEG_channels if c not in bad_channels_here]

    if top_n_channels_input is None:
      top_n_channels = get_channels_with_top_magnitude(meg_data._data[good_meg_channels,:])
      top_n_channels_output['meg'][i] = top_n_channels
    else:
      top_n_channels = top_n_channels_input['meg'][i]
      top_n_channels = [c for c in top_n_channels if c not in bad_channels_here]
    plot_data(meg_data, top_n_channels, ax[i])


  prefix = ""
  if bad_channels:
    prefix += "_bad_channels_removed"
  if denoise:
    prefix += "_denoised"
  fig_name = '/content/my_folder/data/plots/%s_top_meg_channels%s.png' % (subject, prefix)

  fig_meg.savefig(fig_name)

  print('Saving meg figs %s...' % fig_name)
  return fig_meg, fig_eeg, top_n_channels_output


def plot_data(data, channels, ax, every_ms=10, duration=None):
  sample_rate = 1000
  if duration is None:
    duration = int(len(data._data[0,:]) / sample_rate) - 1
    print('duration:')
    print(duration)
  ax.plot(np.arange(0, duration, 1/(sample_rate/every_ms)), np.transpose(data._data[channels,0:duration*sample_rate:every_ms]*1000000)) # uV
  ax.legend([str(i) for i in channels])
  # return fig


def get_channels_with_top_magnitude(data, top_n=10):
  # Consider use peak-to-peak
  abs_max = np.max(np.abs(data), axis=1)
  sort_index = np.argsort(abs_max)
  return sort_index[::-1][:top_n]


def load_bad_channel_data(file):
  df = pd.read_csv(file)
  bad_channels = {}
  for _, row in df.iterrows():
    if row['subject'] not in bad_channels:
      bad_channels[row['subject']] = {}
    if row['task'] not in bad_channels[row['subject']]:
      # Empty entry (None) means eyeballing hasn't been done
      bad_channels[row['subject']][row['task']] = None
    if isinstance(row['bad_channels'],str):
      # Empty list ("[]") means eyeballing has been done and all channels are good
      bad_channels[row['subject']][row['task']] = eval(row['bad_channels'])
  return bad_channels
