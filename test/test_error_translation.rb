# frozen_string_literal: true

require 'test_helper'

class TestErrorTranslation < Minitest::Test
  TRANSLATION = LLMTune::Backend::ErrorTranslation

  def test_maps_each_stage_to_its_error_class
    {
      'dependency' => LLMTune::DependencyError,
      'device' => LLMTune::DependencyError,
      'model_load' => LLMTune::ModelLoadError,
      'dataset' => LLMTune::DatasetError,
      'training' => LLMTune::TrainingError,
      'adapter_save' => LLMTune::TrainingError,
      'merge' => LLMTune::MergeError,
      'generation' => LLMTune::GenerationError
    }.each do |stage, error_class|
      translated = TRANSLATION.translate(RuntimeError.new("[stage:#{stage}] boom"),
                                         default: LLMTune::Error)

      assert_instance_of error_class, translated
      assert_equal 'boom', translated.message
    end
  end

  def test_falls_back_to_the_default_class_without_a_stage_token
    translated = TRANSLATION.translate(RuntimeError.new('boom'), default: LLMTune::TrainingError)

    assert_instance_of LLMTune::TrainingError, translated
    assert_equal 'boom', translated.message
  end

  def test_falls_back_to_the_default_class_for_an_unknown_stage
    translated = TRANSLATION.translate(RuntimeError.new('[stage:mystery] boom'),
                                       default: LLMTune::MergeError)

    assert_instance_of LLMTune::MergeError, translated
  end

  def test_preserves_the_original_error_as_the_cause_when_raised_in_a_rescue
    original = RuntimeError.new('[stage:training] boom')

    translated = begin
      begin
        raise original
      rescue RuntimeError => e
        raise TRANSLATION.translate(e, default: LLMTune::TrainingError)
      end
    rescue LLMTune::TrainingError => e
      e
    end

    assert_same original, translated.cause
  end
end
