For example, a complex instruction can be created as follow:

.. code:: ipython2

    from caqtus.device.sequencer.instructions import (
        Pattern,
        ramp,
        plot_instruction,
    )

    instr = (
        (Pattern([0]) * 2 + Pattern([1])) * 6
        + Pattern([0]) * 5
        + ramp(0.0, 1.5, 20)
        + Pattern([1.5]) * 15
    )

    instr




.. image:: sequencer_instruction_example_files/sequencer_instruction_example_1_0.svg



It represents the following sequence of values:

.. code:: ipython2

    plot_instruction(instr);



.. image:: sequencer_instruction_example_files/sequencer_instruction_example_3_0.png
