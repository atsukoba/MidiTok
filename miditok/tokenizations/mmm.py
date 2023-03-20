from copy import copy
from math import ceil
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange, TimeSignature

from ..classes import Event, TokSequence
from ..constants import (
    ADDITIONAL_TOKENS,
    BEAT_RES,
    CHORD_MAPS,
    MIDI_INSTRUMENTS,
    NB_VELOCITIES,
    PITCH_RANGE,
    SPECIAL_TOKENS,
    TEMPO,
    TIME_DIVISION,
    TIME_SIGNATURE,
)
from ..midi_tokenizer import MIDITokenizer, _in_as_seq, _out_as_complete_seq
from ..utils import detect_chords

DEFAULT_VELOCITY = 80


class MMMTrack(MIDITokenizer):
    r"""MMM (Multi-Track Music Machine, 2020) is extended REMI representation (Huang and Yang) for general
    multi-track, multi-signature symbolic music sequences, introduced in
    `FIGARO (Rütte et al.) <https://arxiv.org/abs/2008.06048>`, which
    represents notes as successions of *Program* (originally *Instrument* in the paper),
    *Pitch*, *Velocity* and *Duration* tokens, and time with *Bar* and *Position* tokens.
    A *Bar* token indicate that a new bar is beginning, and *Position* the current
    position within the current bar. The number of positions is determined by
    the ``beat_res`` argument, the maximum value will be used as resolution.

    :param pitch_range: range of MIDI pitches to use
    :param beat_res: beat resolutions, as a dictionary:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys are tuples indicating a range of beats, ex 0 to 3 for the first bar, and
            the values are the resolution to apply to the ranges, in samples per beat, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: additional tokens (chords, time signature, rests, tempo...) to use,
            to be given as a dictionary. (default: None is used)
    :param special_tokens: list of special tokens. This must be given as a list of strings given
            only the names of the tokens. (default: ``["PAD", "BOS", "EOS", "MASK"]``)
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    :param density_bins: Note density percentile bins.
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[
            str, Union[bool, int, Tuple[int, int]]
        ] = ADDITIONAL_TOKENS,
        programs: Optional[List[int]] = None,
        special_tokens: List[str] = SPECIAL_TOKENS,
        params: Optional[Union[str, Path]] = None,
        density_bins: Optional[List[int]] = None,
        use_velocity: bool = True,
    ):
        self.encoder = []
        additional_tokens["Program"] = True  # required originally Instrument token
        additional_tokens["Rest"] = False
        additional_tokens["Chord"] = False
        additional_tokens["TimeSignature"] = False
        additional_tokens["Tempo"] = False

        # Conditional tokens
        self.programs = list(range(-1, 128)) if programs is None else programs
        self.density_bins = density_bins
        self.use_velocity = use_velocity

        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            special_tokens,
            unique_track=True,  # handles multi-track sequences in single stream
            params=params,  # type: ignore
        )

    def save_params(
        self, out_path: Union[str, Path], additional_attributes: Optional[Dict] = None
    ):
        r"""Saves the config / parameters of the tokenizer in a json encoded file.
        This can be useful to keep track of how a dataset has been tokenized.

        :param out_path: output path to save the file.
        :param additional_attributes: any additional information to store in the config file.
                It can be used to override the default attributes saved in the parent method. (default: None)
        """
        if additional_attributes is None:
            additional_attributes = {}
        additional_attributes_tmp = {
            "density_bins": self.density_bins,
            "use_velocity": self.use_velocity,
            **additional_attributes,
        }
        super().save_params(out_path, additional_attributes_tmp)

    @_out_as_complete_seq
    def _midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> TokSequence:
        r"""Override the parent class method
        Converts a MIDI file in a token representation
        :param midi: the MIDI object to convert
        :return: sequences of tokens
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self._durations_ticks:
            self._durations_ticks[midi.ticks_per_beat] = np.array(
                [
                    (beat * res + pos) * midi.ticks_per_beat // res
                    for beat, pos, res in self.durations
                ]
            )

        # Preprocess the MIDI file
        self.preprocess_midi(midi)

        # Register MIDI metadata
        self.current_midi_metadata = {
            "time_division": midi.ticks_per_beat,
            "tempo_changes": midi.tempo_changes,
            "time_sig_changes": midi.time_signature_changes,
            "key_sig_changes": midi.key_signature_changes,
        }

        # **************** OVERRIDE FROM HERE, KEEP THE LINES ABOVE IN YOUR METHOD ****************

        # Convert each track to tokens
        events: List[Event] = [Event(type="Piece", value="Start", time=0, desc="None")]
        for track in midi.instruments:
            if track.program in self.programs:
                events += cast(List[Event], self.track_to_tokens(track).events)

        return TokSequence(events=cast(List[Union[Event, List[Event]]], events))

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens (:class:`miditok.TokSequence`).
        (can probably be achieved faster with Mido objects)

        :param track: MIDI track to convert
        :return: :class:`miditok.TokSequence` of corresponding tokens.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_bar = self._current_midi_metadata["time_division"] * 4
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        events: List[Event] = [
            Event(
                type="Track",
                value="Start",
                time=0,
                desc=MIDI_INSTRUMENTS[track.program] if not track.is_drum else "Drum",
            )
        ]

        # Creates the Bar, Note On, Note Off and Velocity events
        previous_tick = 0
        current_bar = -1
        for note in track.notes:
            if note.start != previous_tick:
                # Bar
                nb_new_bars = note.start // ticks_per_bar - current_bar
                for i in range(nb_new_bars):
                    if current_bar != -1:
                        events.append(
                            Event(
                                type="Bar",
                                value="End",
                                time=(current_bar + i) * ticks_per_bar,
                                desc=0,
                            )
                        )
                    events.append(
                        Event(
                            type="Bar",
                            value="Start",
                            time=(current_bar + i + 1) * ticks_per_bar,
                            desc=0,
                        )
                    )

                current_bar += nb_new_bars
            # Note On
            events.append(
                Event(type="NoteOn", value=note.pitch, time=note.start, desc=note.end)
            )
            # Velocity
            if self.use_velocity:
                events.append(
                    Event(
                        type="Velocity",
                        value=note.velocity,
                        time=note.start,
                        desc=f"{note.velocity}",
                    )
                )
            # Note Off
            events.append(
                Event(type="NoteOff", value=note.pitch, time=note.end, desc=note.end)
            )

        # Sorts events
        events.sort(
            key=lambda e: (e.time, self._order(e))
        )  # Sort by time then track then pitch)

        # Time Shift
        previous_tick = 0
        previous_note_end = track.notes[0].start + 1
        for e, event in enumerate(events.copy()):
            # No time shift
            if event.time == previous_tick:
                pass

            # TimeShift
            else:
                time_shift = event.time - previous_tick
                index = np.argmin(np.abs(dur_bins - time_shift))
                events.append(
                    Event(
                        type="TimeShift",
                        value=".".join(map(str, self.durations[index])),
                        time=previous_tick,
                        desc=f"{time_shift} ticks",
                    )
                )

            if event.type == "NoteOn":
                previous_note_end = max(previous_note_end, event.desc)
            previous_tick = event.time

        events.sort(key=lambda x: (x.time, self._order(x)))
        events.append(
            Event(
                type="Track",
                value="End",
                time=previous_tick,
                desc=MIDI_INSTRUMENTS[track.program] if not track.is_drum else "Drum",
            )
        )

        return TokSequence(events=cast(List[Union[Event, List[Event]]], events))

    @_in_as_seq()
    def tokens_to_track(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        time_division: int = TIME_DIVISION,
        program: Tuple[int, bool] = (0, False),
        default_duration: Optional[int] = None,
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :param default_duration: default duration (in ticks) in case a Note On event occurs without its associated
                                note off event. Leave None to discard Note On with no Note Off event.
        :return: the miditoolkit instrument object and tempo changes
        """
        ticks_per_sample = time_division // max(self.beat_res.values())
        tokens = cast(TokSequence, tokens)
        events = (
            cast(List[Event], tokens.events)
            if tokens.events is not None
            else list(map(lambda tok: Event(*tok.split("_")), tokens.tokens))
        )

        max_duration = self.durations[-1][0] * time_division + self.durations[-1][1] * (
            time_division // self.durations[-1][2]
        )
        name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [
            TempoChange(TEMPO, -1)
        ]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        while ei < len(events):
            if events[ei].type == "NoteOn":
                try:
                    if (
                        events[ei + 1].type == "Velocity"
                        or events[ei + 1].type == "TimeShift"
                    ):
                        pitch = int(events[ei].value)
                        vel = int(
                            events[ei + 1].value
                            if events[ei + 1].type == "Velocity"
                            else DEFAULT_VELOCITY
                        )

                        # look for an associated note off event to get duration
                        offset_tick = 0
                        duration = 0
                        for i in range(ei + 1, len(events)):
                            if (
                                events[i].type == "NoteOff"
                                and int(events[i].value) == pitch
                            ):
                                duration = offset_tick
                                break
                            elif events[i].type == "TimeShift":
                                offset_tick += self._token_duration_to_ticks(
                                    str(events[i].value), time_division
                                )
                            if (
                                offset_tick > max_duration
                            ):  # will not look for Note Off beyond
                                break

                        if duration == 0 and default_duration is not None:
                            duration = default_duration
                        if duration != 0:
                            instrument.notes.append(
                                Note(vel, pitch, current_tick, current_tick + duration)
                            )
                        ei += 1
                except IndexError:
                    pass
            elif events[ei].type == "TimeShift":
                current_tick += self._token_duration_to_ticks(
                    str(events[ei].value), time_division
                )
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_base_vocabulary(self) -> List[str]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = ["Piece_Start"]

        # Track
        vocab += ["Track_Start", "Track_End"]

        # PROGRAM
        if self.additional_tokens["Program"]:
            vocab += [f"Program_{program}" for program in range(-1, 128)]

        # BAR
        vocab += ["Bar_Start", "Bar_End"]

        # NOTE ON
        vocab += [f"NoteOn_{i}" for i in self.pitch_range]

        # VELOCITY
        if self.use_velocity:
            vocab += [f"Velocity_{i}" for i in self.velocities]
        
        # NOTE OFF
        vocab += [f"NoteOff_{i}" for i in self.pitch_range]

        # TIME SHIFTS
        vocab += [
            f'TimeShift_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # DENSITY
        if self.density_bins is not None:
            vocab += [f"Density_{v}" for v in range(1, len(self.density_bins))]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["Piece"] = ["Track"]
        dic["Track"] = ["Program", "Track"]
        if self.density_bins:
            dic["Program"] = ["Density"]
            dic["Density"] = ["Bar"]
        else:
            dic["Program"] = ["Bar"]
        dic["Bar"] = ["NoteOn", "TimeShift", "Bar"]
        if self.use_velocity:
            dic["NoteOn"] = ["Velocity"]
            dic["Velocity"] = ["NoteOn", "TimeShift"]
        else:
            dic["NoteOn"] = ["NoteOn", "TimeShift"]
        dic["TimeShift"] = ["NoteOff"]
        dic["NoteOff"] = ["NoteOff", "NoteOn", "TimeShift"]
        return dic

    @_in_as_seq(complete=False, decode_bpe=False)
    def tokens_errors(self, tokens: Union[TokSequence, List, np.ndarray, Any]) -> float:
        r"""Checks if a sequence of tokens is made of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a NoteOn token should not be present if the same pitch is already being played
            - a NoteOff token should not be present the note is not being played

        :param tokens: sequence of tokens to check
        :return: the error ratio (lower is better)
        """
        tokens = cast(TokSequence, tokens)
        nb_tok_predicted = len(tokens)  # used to norm the score
        if self.has_bpe:
            self.decode_bpe(tokens)
        self.complete_sequence(tokens)

        err = 0
        current_pitches = []
        max_duration = self.durations[-1][0] * max(self.beat_res.values())
        max_duration += self.durations[-1][1] * (
            max(self.beat_res.values()) // self.durations[-1][2]
        )

        events = (
            cast(List[Event], tokens.events)
            if tokens.events is not None
            else list(map(lambda tok: Event(*tok.split("_")), tokens.tokens))
        )

        for i in range(1, len(events)):
            # Good token type
            if events[i].type in self.tokens_types_graph[events[i - 1].type]:
                if events[i].type == "NoteOn":
                    if int(events[i].value) in current_pitches:
                        err += 1  # pitch already being played
                        continue

                    current_pitches.append(int(events[i].value))
                    # look for an associated note off event to get duration
                    offset_sample = 0
                    for j in range(i + 1, len(events)):
                        if events[j].type == "NoteOff" and int(events[j].value) == int(
                            events[i].value
                        ):
                            break  # all good
                        elif events[j].type == "TimeShift":
                            offset_sample += self._token_duration_to_ticks(
                                str(events[j].value), max(self.beat_res.values())
                            )

                        if (
                            offset_sample > max_duration
                        ):  # will not look for Note Off beyond
                            err += 1
                            break
                elif events[i].type == "NoteOff":
                    if int(events[i].value) not in current_pitches:
                        err += 1  # this pitch wasn't being played
                    else:
                        current_pitches.remove(int(events[i].value))

            # Bad token type
            else:
                err += 1

        return err / nb_tok_predicted

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Piece":
            return 0
        elif x.type == "Track":
            return 1
        elif x.type == "Program":
            return 2
        elif x.type == "Density":
            return 3
        elif x.type == "Bar" and x.value == "Start":
            return 4
        elif x.type == "Bar" and x.value == "End":
            return 5
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 100


class MMMBar(MIDITokenizer):
    r"""MMMBar model representation (Multi-Track Music Machine, 2020)
    multi-track, multi-signature symbolic music sequences, introduced in
    `FIGARO (Rütte et al.) <https://arxiv.org/abs/2008.06048>`, which
    represents notes as successions of *Program* (originally *Instrument* in the paper),
    *Pitch*, *Velocity* and *Duration* tokens, and time with *Bar* and *Position* tokens.
    A *Bar* token indicate that a new bar is beginning, and *Position* the current
    position within the current bar. The number of positions is determined by
    the ``beat_res`` argument, the maximum value will be used as resolution.

    :param pitch_range: range of MIDI pitches to use
    :param beat_res: beat resolutions, as a dictionary:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys are tuples indicating a range of beats, ex 0 to 3 for the first bar, and
            the values are the resolution to apply to the ranges, in samples per beat, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: additional tokens (chords, time signature, rests, tempo...) to use,
            to be given as a dictionary. (default: None is used)
    :param special_tokens: list of special tokens. This must be given as a list of strings given
            only the names of the tokens. (default: ``["PAD", "BOS", "EOS", "MASK"]``)
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    :param density_bins: Note density percentile bins.
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[
            str, Union[bool, int, Tuple[int, int]]
        ] = ADDITIONAL_TOKENS,
        special_tokens: List[str] = SPECIAL_TOKENS,
        params: Optional[Union[str, Path]] = None,
        density_bins: Optional[List[int]] = None,
    ):
        self.encoder = []
        additional_tokens["Rest"] = False
        additional_tokens["Chord"] = False
        additional_tokens["TimeSignature"] = False
        additional_tokens["Tempo"] = False

        # Conditional tokens
        self.density_bins = density_bins

        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            special_tokens,
            unique_track=False,  # handles multi-track sequences in single stream
            params=params,  # type: ignore
        )

    def save_params(
        self, out_path: Union[str, Path], additional_attributes: Optional[Dict] = None
    ):
        r"""Saves the config / parameters of the tokenizer in a json encoded file.
        This can be useful to keep track of how a dataset has been tokenized.

        :param out_path: output path to save the file.
        :param additional_attributes: any additional information to store in the config file.
                It can be used to override the default attributes saved in the parent method. (default: None)
        """
        if additional_attributes is None:
            additional_attributes = {}
        additional_attributes_tmp = {
            "density_bins": self.density_bins,
            **additional_attributes,
        }
        super().save_params(out_path, additional_attributes_tmp)

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens (:class:`miditok.TokSequence`).
        (can probably be achieved faster with Mido objects)

        :param track: MIDI track to convert
        :return: :class:`miditok.TokSequence` of corresponding tokens.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self._current_midi_metadata["time_division"] / max(
            self.beat_res.values()
        )
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        min_rest = (
            self._current_midi_metadata["time_division"] * self.rests[0][0]
            + ticks_per_sample * self.rests[0][1]
            if self.additional_tokens["Rest"]
            else 0
        )
        events: List[Event] = []

        # Creates the Note On, Note Off and Velocity events
        for note in track.notes:
            # Note On
            events.append(
                Event(type="NoteOn", value=note.pitch, time=note.start, desc=note.end)
            )
            # Velocity
            events.append(
                Event(
                    type="Velocity",
                    value=note.velocity,
                    time=note.start,
                    desc=f"{note.velocity}",
                )
            )
            # Note Off
            events.append(
                Event(type="NoteOff", value=note.pitch, time=note.end, desc=note.end)
            )

        # Sorts events
        events.sort(
            key=lambda e: (e.time, self._order(e))
        )  # Sort by time then track then pitch)

        # Time Shift
        previous_tick = 0
        previous_note_end = track.notes[0].start + 1
        for e, event in enumerate(events.copy()):
            # No time shift
            if event.time == previous_tick:
                pass

            # TimeShift
            else:
                time_shift = event.time - previous_tick
                index = np.argmin(np.abs(dur_bins - time_shift))
                events.append(
                    Event(
                        type="TimeShift",
                        value=".".join(map(str, self.durations[index])),
                        time=previous_tick,
                        desc=f"{time_shift} ticks",
                    )
                )

            if event.type == "NoteOn":
                previous_note_end = max(previous_note_end, event.desc)
            previous_tick = event.time

        events.sort(key=lambda x: (x.time, self._order(x)))

        return TokSequence(events=cast(List[Union[Event, List[Event]]], events))

    @_in_as_seq()
    def tokens_to_track(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        time_division: int = TIME_DIVISION,
        program: Tuple[int, bool] = (0, False),
        default_duration: Optional[int] = None,
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :param default_duration: default duration (in ticks) in case a Note On event occurs without its associated
                                note off event. Leave None to discard Note On with no Note Off event.
        :return: the miditoolkit instrument object and tempo changes
        """
        ticks_per_sample = time_division // max(self.beat_res.values())
        tokens = cast(TokSequence, tokens)
        events = (
            cast(List[Event], tokens.events)
            if tokens.events is not None
            else list(map(lambda tok: Event(*tok.split("_")), tokens.tokens))
        )

        max_duration = self.durations[-1][0] * time_division + self.durations[-1][1] * (
            time_division // self.durations[-1][2]
        )
        name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [
            TempoChange(TEMPO, -1)
        ]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        while ei < len(events):
            if events[ei].type == "NoteOn":
                try:
                    if events[ei + 1].type == "Velocity":
                        pitch = int(events[ei].value)
                        vel = int(events[ei + 1].value)

                        # look for an associated note off event to get duration
                        offset_tick = 0
                        duration = 0
                        for i in range(ei + 1, len(events)):
                            if (
                                events[i].type == "NoteOff"
                                and int(events[i].value) == pitch
                            ):
                                duration = offset_tick
                                break
                            elif events[i].type == "TimeShift":
                                offset_tick += self._token_duration_to_ticks(
                                    str(events[i].value), time_division
                                )
                            if (
                                offset_tick > max_duration
                            ):  # will not look for Note Off beyond
                                break

                        if duration == 0 and default_duration is not None:
                            duration = default_duration
                        if duration != 0:
                            instrument.notes.append(
                                Note(vel, pitch, current_tick, current_tick + duration)
                            )
                        ei += 1
                except IndexError:
                    pass
            elif events[ei].type == "TimeShift":
                current_tick += self._token_duration_to_ticks(
                    str(events[ei].value), time_division
                )
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_base_vocabulary(self) -> List[str]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = []

        # NOTE ON
        vocab += [f"NoteOn_{i}" for i in self.pitch_range]

        # NOTE OFF
        vocab += [f"NoteOff_{i}" for i in self.pitch_range]

        # VELOCITY
        vocab += [f"Velocity_{i}" for i in self.velocities]

        # TIME SHIFTS
        vocab += [
            f'TimeShift_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # FILL
        vocab += [f"Fill_{v}" for v in ["Start", "End", "In"]]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["NoteOn"] = ["NoteOn", "TimeShift"]
        dic["TimeShift"] = ["NoteOff"]
        dic["NoteOff"] = ["NoteOff", "NoteOn", "TimeShift"]
        return dic

    @_in_as_seq(complete=False, decode_bpe=False)
    def tokens_errors(self, tokens: Union[TokSequence, List, np.ndarray, Any]) -> float:
        r"""Checks if a sequence of tokens is made of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a NoteOn token should not be present if the same pitch is already being played
            - a NoteOff token should not be present the note is not being played

        :param tokens: sequence of tokens to check
        :return: the error ratio (lower is better)
        """
        tokens = cast(TokSequence, tokens)
        nb_tok_predicted = len(tokens)  # used to norm the score
        if self.has_bpe:
            self.decode_bpe(tokens)
        self.complete_sequence(tokens)

        err = 0
        current_pitches = []
        max_duration = self.durations[-1][0] * max(self.beat_res.values())
        max_duration += self.durations[-1][1] * (
            max(self.beat_res.values()) // self.durations[-1][2]
        )

        events = (
            cast(List[Event], tokens.events)
            if tokens.events is not None
            else list(map(lambda tok: Event(*tok.split("_")), tokens.tokens))
        )

        for i in range(1, len(events)):
            # Good token type
            if events[i].type in self.tokens_types_graph[events[i - 1].type]:
                if events[i].type == "NoteOn":
                    if int(events[i].value) in current_pitches:
                        err += 1  # pitch already being played
                        continue

                    current_pitches.append(int(events[i].value))
                    # look for an associated note off event to get duration
                    offset_sample = 0
                    for j in range(i + 1, len(events)):
                        if events[j].type == "NoteOff" and int(events[j].value) == int(
                            events[i].value
                        ):
                            break  # all good
                        elif events[j].type == "TimeShift":
                            offset_sample += self._token_duration_to_ticks(
                                str(events[j].value), max(self.beat_res.values())
                            )

                        if (
                            offset_sample > max_duration
                        ):  # will not look for Note Off beyond
                            err += 1
                            break
                elif events[i].type == "NoteOff":
                    if int(events[i].value) not in current_pitches:
                        err += 1  # this pitch wasn't being played
                    else:
                        current_pitches.remove(int(events[i].value))

            # Bad token type
            else:
                err += 1

        return err / nb_tok_predicted

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Piece":
            return 0
        elif x.type == "Track":
            return 1
        elif x.type == "Program":
            return 2
        elif x.type == "Density":
            return 3
        elif x.type == "Bar" and x.value == "Start":
            return 4
        elif x.type == "Bar" and x.value == "End":
            return 5
        elif x.type == "NoteOn":
            return 6
        elif x.type == "TimeShift":
            return 7
        elif x.type == "NoteOff":
            return 8
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 9
