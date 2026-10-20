#!/usr/bin/python3
from __future__ import annotations

import typing

from bbq_level import BBQLevel
from bbq_level_pb import BBQLevelPB
from codegen import CodeGen

# Hack for type hinting with circular imports
if typing.TYPE_CHECKING: from bbq import BBQ


class BBQLevelLX(BBQLevel):
    """Represents an LX BBQ level."""
    def __init__(self, bbq: BBQ, start_cycle: int, level_id: int,
                 sram_bitmap: bool, sram_counters: bool) -> None:

        super().__init__(bbq, start_cycle)

        self.level_id = level_id            # BBQ Level (>= 1)
        self.sram_bitmap = sram_bitmap      # Store bitmap in SRAM?
        self.sram_counters = sram_counters  # Store counters in SRAM?


    @property
    def is_leaf(self) -> bool:
        """Returns whether this is a leaf bitmap."""
        return isinstance(self.next_level, BBQLevelPB)


    @property
    def id(self) -> int:
        """Alias for the level ID."""
        return self.level_id


    def name(self) -> str:
        """Canonical level name."""
        return "l{}".format(self.level_id)


    def latency(self) -> int:
        """Latency in cycles."""
        return 1 + int(self.sram_bitmap) + int(self.sram_counters)


    def emit_stage_defs(self, cg: CodeGen) -> None:
        """Emit this level through the supplied CodeGen instance."""
        raise NotImplementedError("Bitmap-level implementation removed")


    def emit_state_dependent_combinational_logic(self, cg: CodeGen) -> None:
        """Emit this level through the supplied CodeGen instance."""
        raise NotImplementedError("Bitmap-level implementation removed")


    def emit_combinational_default_assigns(self, cg: CodeGen) -> None:
        """Emit this level through the supplied CodeGen instance."""
        raise NotImplementedError("Bitmap-level implementation removed")


    def emit_state_agnostic_combinational_logic(self, cg: CodeGen) -> None:
        """Emit this level through the supplied CodeGen instance."""
        raise NotImplementedError("Bitmap-level implementation removed")


    def emit_sequential_pipeline_logic(self, cg: CodeGen) -> None:
        """Emit this level through the supplied CodeGen instance."""
        raise NotImplementedError("Bitmap-level implementation removed")
