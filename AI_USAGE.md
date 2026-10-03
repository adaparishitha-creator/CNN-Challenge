                                                            AI Usage
1)AI Tool Used:-

I used ChatGPT as AI tool for this assignment. I used it to learn about different CNN architectures and to help with some code implementations. I also used it while debugging my code. Whenever my code was failing, ChatGPT helped me understand the possible reasons for failure and what requirements or dependencies that might be missing.

2)Examples of AI Assistance:-

AI helped me a lot in understanding different types of CNN architectures available. Even within the same architecture, there are different variants such as Tiny, Small, and Base, and ChatGPT helped me understand these different options. When I wanted to explore scene-specific pretraining, it also helped me identifying datasets related to scene recognition, such as Places365, MIT scene datasets, and SUN, and suggested Places365 as the most relevant option for my task among the ones I considered. AI also helped me with some code implementations, especially for data augmentation techniques such as MixUp, CutMix, and Cutout.

3)Ineffective AI Suggestion:-

One suggestion from AI was to try ConvNeXtV2-Base when I want to experiment with a newer CNN architecture. AI suggested it as recent and powerful CNN architecture that could be worth testing. It gave good accuracy in my experiment, but for my specific image classification task, it did not perform as well as ConvNeXt-Small. This showed me that using a newer or larger architecture does not always necessarily give better results for every dataset.

4)How I Verified or Modified AI-Generated Solutions:-

I did not blindly rely on the solutions or suggestions provided by AI. Whatever suggestions I received from AI, I manually implemented and ran the experiments on my system. I then compared the accuracies across different experiments to check whether the suggested approaches were actually improving or decreasing the performance. In this way, I verified the AI-generated solutions based on actual experimental results rather than assuming the suggestions were correct.

5)Important Decision I Made:-

An important decision I made was to choose the final architecture based on my own experimental results. Throughout the assignment, I studied different CNN architectures and decided which architecture to try next based on what I learned from previous experiments. After getting good accuracy with Experiment 6, I checked with AI to see if there were any newer or more powerful CNN architectures that could potentially perform better. AI suggested ConvNeXtV2-Base, which I tested in Experiment 7. However, after comparing the experimental results, I found that Experiment 6 using ConvNeXt-Small pretrained on ImageNet-22K with advanced augmentation performed better for this specific task. Therefore, I decided to select Experiment 6 as my final model based on the actual experimental performance rather than simply selecting architecture which was suggested by AI.


