# frozen_string_literal: true

require 'test_helper'

class TestErrorTranslation < Minitest::Test
  TRANSLATION = Grain::Backend::ErrorTranslation

  def test_maps_each_stage_to_its_error_class
    {
      'dependency' => Grain::DependencyError,
      'device' => Grain::DependencyError,
      'model_load' => Grain::ModelLoadError,
      'dataset' => Grain::DatasetError,
      'training' => Grain::TrainingError,
      'adapter_save' => Grain::TrainingError,
      'merge' => Grain::MergeError,
      'generation' => Grain::GenerationError
    }.each do |stage, error_class|
      translated = TRANSLATION.translate(RuntimeError.new("[stage:#{stage}] boom"),
                                         default: Grain::Error)

      assert_instance_of error_class, translated
      assert_equal 'boom', translated.message
    end
  end

  def test_falls_back_to_the_default_class_without_a_stage_token
    translated = TRANSLATION.translate(RuntimeError.new('boom'), default: Grain::TrainingError)

    assert_instance_of Grain::TrainingError, translated
    assert_equal 'boom', translated.message
  end

  def test_falls_back_to_the_default_class_for_an_unknown_stage
    translated = TRANSLATION.translate(RuntimeError.new('[stage:mystery] boom'),
                                       default: Grain::MergeError)

    assert_instance_of Grain::MergeError, translated
  end

  def test_preserves_the_original_error_as_the_cause_when_raised_in_a_rescue
    original = RuntimeError.new('[stage:training] boom')

    translated = begin
      begin
        raise original
      rescue RuntimeError => e
        raise TRANSLATION.translate(e, default: Grain::TrainingError)
      end
    rescue Grain::TrainingError => e
      e
    end

    assert_same original, translated.cause
  end
end
