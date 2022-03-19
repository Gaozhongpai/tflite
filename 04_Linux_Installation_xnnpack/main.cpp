#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

int main()
{
    auto m_model = TfLiteModelCreateFromFile("../../models/hand_mesh.tflite");
    auto m_options = TfLiteInterpreterOptionsCreate();
    TfLiteXNNPackDelegateOptions opt = TfLiteXNNPackDelegateOptionsDefault();
    auto m_xnnpack_delegate = TfLiteXNNPackDelegateCreate(&opt);
    TfLiteInterpreterOptionsAddDelegate(m_options, m_xnnpack_delegate);
    auto m_interpreter = TfLiteInterpreterCreate(m_model, m_options); // This returns nullptr and prints error messages in the console
}
