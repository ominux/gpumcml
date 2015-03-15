

The following list of FAQs is gathered from our development team's experience and from other user's feedback.  Feel free to let us know what troubles you are having.

### Problem 1: No CUDA Compatible Device Found ###
**Solution**: Install the latest drivers from NVIDIA.  Don't use the default Windows Driver that is automatically installed at start-up immediately after installing the new card.

### Problem 2: My screen freezes ###
**Solution**: Use a dedicated GPU to run your simulation, if possible.  There is a known time-out issue when a GPU is attached to the monitor.   As an alternative, go to **gpumcml\_kernel.h** and reduce **NUM\_STEPS** as this parameter depends on how fast your graphics card is.

```
#define NUM_STEPS 50000
```