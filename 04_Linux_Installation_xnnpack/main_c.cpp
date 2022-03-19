#include "tensorflow/lite/c/c_api.h"
#include <stdio.h>
#include <stdlib.h>


int main (int argc, char* argv[]) {

   for(int i = 0; i < 3; i++)
    {
        printf("Iteration: %d\n", i);

        float input[49] = { 0.0 };
        TfLiteModel* model = TfLiteModelCreateFromFile("../../models/model.fp32.tflite");
        TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(options, 2);
        TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
        TfLiteInterpreterAllocateTensors(interpreter);

        TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        TfLiteTensorCopyFromBuffer(input_tensor, input, 49 * sizeof(float));

        TfLiteInterpreterInvoke(interpreter);

        const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 14);

        float output[49];
        TfLiteTensorCopyToBuffer(output_tensor, output, 49 * sizeof(float));

        printf("Output: \n\n");
        for (int j = 0; j < 49; j++) {
            printf("%d: %f\n", j, output[j]);
        }

        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
    }
    return 0;
}