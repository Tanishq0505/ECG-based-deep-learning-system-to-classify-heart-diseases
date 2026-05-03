import streamlit as st
from Processing import  ECG
ecg = ECG()
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  try:
    ecg_user_image_read = ecg.getImage(uploaded_file)
    st.image(ecg_user_image_read)

    ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
    
    my_expander = st.expander(label='Gray SCALE IMAGE')
    with my_expander: 
      st.image(ecg_user_gray_image_read)
    
    dividing_leads=ecg.DividingLeads(ecg_user_image_read)

    my_expander1 = st.expander(label='DIVIDING LEAD')
    with my_expander1:
      st.image('Leads_1-12_figure.png')
      st.image('Long_Lead_13_figure.png')
    
    ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)

    my_expander2 = st.expander(label='PREPROCESSED LEAD')
    with my_expander2:
      st.image('Preprossed_Leads_1-12_figure.png')
      st.image('Preprossed_Leads_13_figure.png')
    
    ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
    my_expander3 = st.expander(label='CONOTUR LEADS')
    with my_expander3:
      st.image('Contour_Leads_1-12_figure.png')
    
    ecg_1dsignal = ecg.CombineConvert1Dsignal()
    my_expander4 = st.expander(label='1D Signals')
    with my_expander4:
      st.write(ecg_1dsignal)
      
    ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
    my_expander4 = st.expander(label='Dimensional Reduction')
    with my_expander4:
      st.write(ecg_final)
    
    ecg_model=ecg.ModelLoad_predict(ecg_final)
    my_expander5 = st.expander(label='PREDICTION')
    with my_expander5:
      st.write(ecg_model)
  except ValueError as error:
    st.error(f"Invalid or unsupported ECG image: {error}")
    st.info("Upload a clear 12-lead ECG report image (full page, not cropped, with visible waveforms).")
  except Exception as error:
    st.error(f"Processing failed: {error}")
