
import librosa
import soundfile
from scipy.io import wavfile
import os
import mne
from fast_align_audio import alignment
import matplotlib.pyplot as plt
import matplotlib
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

# min_speech_segment_gap: second. Any two segments separated less than this should be grouped into a single segment
# min_speech_segment_length: second. Any utterance should be longer than this
MIN_SPEECH_SEGMENT_GAP = 1.0
MIN_SPEECH_SEGMENT_LENGTH = 0.1

# https://github.com/pyannote/pyannote-audio
vad_pipeline = Pipeline.from_pretrained(
    # "pyannote/voice-activity-detection",
    "pyannote/speaker-diarization-3.0",
    use_auth_token="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")

print(vad_pipeline)


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

def get_data_for_subject_task(subject_MEG, task, local_dir = "./my_folder/"):
  # subject_MEG: e.g., "G02"

  subject_dirs = _get_subject_dirs(subject_MEG, local_dir)
  print(subject_dirs)

  fname_raw = glob.glob(subject_dirs["meg_dir"] + "*" + task + ".con")[0]
  # fname_elp = glob.glob(meg_dir + "*.elp")
  # fname_hsp = glob.glob(meg_dir + "*.hsp")
  # fname_mrk = glob.glob(meg_dir + "*.mrk")

  # .WAV audio data
  interviewer_audio = glob.glob(subject_dirs["audio_dir"] + "console_mic*" + task + ".wav")[0]
  participant_audio = glob.glob(subject_dirs["audio_dir"] + "subject_mic*" + task + ".wav")[0]

  return {"meg_file":fname_raw, 
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
  recognizer = speech_recognition.Recognizer()
  speech_texts = []
  for seg_index, seg in enumerate(segments):
    seg_audio_file = audio_file.replace(".wav", f"_seg_{seg_index:0>4}.wav")
    with speech_recognition.AudioFile(seg_audio_file) as source:
      try:
        audio_seg = recognizer.record(source)
        t_current = seg[1]
        # speech_text = recognizer.recognize_google(audio_seg)
        speech_text = recognizer.recognize_whisper_api(audio_seg)
        # recognize_sphinx
        # recognize_wit
        print(f"[Segment #{seg_index:0>4}] between {seg[0]:.3f} and {seg[1]:.3f}:")
        print(speech_text)
        speech_texts.append(speech_text)
      except:
        # print(' ')
        speech_texts.append('')
        continue
    return speech_texts



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
    else:
      segments = _segement_speech(left_channel_audio_file, plot=plot)
      postprocess_and_save_segment(segments, audio_file)

    # _export_audio_segments(segments, left_channel_audio_file)

    # speech_texts = _transcribe_speech(segments, left_channel_audio_file)


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
  # example: load_processed_audio_data("G02", "B1", inspect_alignment=True, inspect_segmentation=True)
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

  for prefix, audio_file, meg_audio_file in [("interviewer_", interviewer_audio_file_raw, interviewer_audio_file_meg),("participant_", participant_audio_file_raw, participant_audio_file_meg)]:
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
   
