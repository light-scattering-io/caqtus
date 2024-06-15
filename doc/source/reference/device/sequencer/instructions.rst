caqtus.device.sequencer.instructions
====================================

.. module:: caqtus.device.sequencer.instructions

   This module contains the implementation of :class:`SequencerInstruction` and its subclasses.
   These classes are used to represent sequences of values to output on several channels in a compact form.

   Usage
   ~~~~~

   .. include:: example.rst

   Classes
   ~~~~~~~

    .. inheritance-diagram:: Pattern Concatenated Repeated Ramp
       :parts: 1
       :top-classes: SequencerInstruction

    .. autoclass:: SequencerInstruction

    .. autoclass:: Pattern

    .. autoclass:: Concatenated

    .. autoclass:: Repeated

    .. autoclass:: Ramp

   Functions
   ~~~~~~~~~

   .. autofunction:: concatenate

   .. autofunction:: ramp

   .. autofunction:: with_name

   .. autofunction:: stack_instructions

   .. autofunction:: plot_instruction

   .. autofunction:: to_graph
