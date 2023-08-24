# %%
import os

from SRTmodel import (
    auto_generate_subtitle,
    auto_generate_subtitle_NashVersion,
    auto_generate_subtitle_preprocessing,
    long_speech_recognition_extra_function,
    move_file,
    punctuation_auto_generate_subtitle_preprocessing,
    punctuation_auto_generate_subtitle_second_adj,
    video2text_basic,
)

# %%
folder = '影片'
video_name_list = list(filter(lambda x: 'wav' not in x, os.listdir(folder)))

for video_name in video_name_list:
    video_path = folder + '//' + video_name
    video2text_basic(video_path)

# %%

for video_name in video_name_list:  # i = video_list[0]
    # 要先指定folder

    # 要給影片名稱
    video = video_name
    video = video.split('.')[0]

    # 語音辨識
    from google.cloud import storage

    client = storage.Client.from_service_account_json(
        "gcpvideo2text-ac3ac35877e6.json"
    )  # 注意要修改的金鑰
    response = long_speech_recognition_extra_function(
        # 環境參數設定
        json='gcpvideo2text-ac3ac35877e6.json',  # 注意要修改的金鑰
        speech_path=os.getcwd() + '//' + folder + '//' + video + '.wav',
        # 音檔設定
        sample_rate_hertz=48000,
        language_code="cmn-Hant-TW",
        bucket_name='gcp_storage_v1',  # 注意要修改的bucket_name
        destination_blob_name=video + '.wav',
        audio_type='WAV',
        # 辨識功能
        enable_word_confidence=True,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        # 區分說話人員
        enable_speaker_diarization=False,
        audio_channel_count=1,
        diarization_speaker_count=1,
        # 音檔修改專用
        speech_contexts=[''],
        # 產出文件
        enable_docx=True,
        enable_xlsx=True,
        enable_wordcloud=True,
    )

    # 首先找出word_table
    word_table, auto_select_range = auto_generate_subtitle_preprocessing(
        response=response, sec_split=2
    )

    # 句讀切割
    (
        new_word_table,
        auto_select_range,
    ) = punctuation_auto_generate_subtitle_preprocessing(
        response=response, word_table=word_table
    )

    # 句讀 + 秒數切割
    # 本處的sec代表句讀太長的字句超過多少秒要執行切分
    # 這邊預設5秒
    (
        new_word_table,
        auto_select_range,
    ) = punctuation_auto_generate_subtitle_second_adj(
        new_word_table=new_word_table, sec=5
    )

    # 如果最後一個標點符號是「，」，則會被辨識成cut，但是這是不對的，所以若最後一個 == cut
    # 我就要將它變回「no」
    if new_word_table['cut_para'].iloc[-1] == 'cut':
        new_word_table['cut_para'].iloc[-1] = 'no'
        auto_select_range = auto_select_range[:-1]

    # 產出字幕
    srt_list = auto_generate_subtitle(
        word_table=new_word_table,
        auto_select_range=auto_select_range,
        video=video + '_句讀與秒數切割',
    )

    move_file(dectect_name=video, folder_name=video)

# %%
