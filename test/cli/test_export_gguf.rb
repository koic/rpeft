# frozen_string_literal: true

require 'test_helper'

class TestCLIExportGGUF < RPEFTTest
  def setup
    super
    @workdir = Dir.mktmpdir('rpeft_test')
    @model_dir = File.join(@workdir, 'model')
    FileUtils.mkdir_p(@model_dir)
    File.write(File.join(@model_dir, 'config.json'), %({"model_type":"qwen2"}))
    @llama_cpp = File.join(@workdir, 'llama.cpp')
    FileUtils.mkdir_p(@llama_cpp)
    File.write(File.join(@llama_cpp, 'convert_hf_to_gguf.py'), "# converter\n")
    @output = File.join(@workdir, 'model.gguf')
    @commands = []
    RPEFT::GGUF.runner = lambda do |argv, _env|
      @commands << argv
      File.binwrite(argv[argv.index('--outfile') + 1], 'GGUF')
      ['', true]
    end
  end

  def teardown
    RPEFT::GGUF.runner = nil
    FileUtils.remove_entry(@workdir)
    super
  end

  def test_exports_and_reports_the_gguf_path
    status, out, = run_cli(['export', 'gguf', '--model', @model_dir, '--output', @output, '--llama-cpp', @llama_cpp])

    assert_equal 0, status
    assert_includes out, "GGUF saved to #{@output}"
    assert_equal 'q8_0', @commands.first.last
  end

  def test_passes_the_quantize_type
    run_cli(['export', 'gguf', '--model', @model_dir, '--output', @output, '--llama-cpp', @llama_cpp, '--quantize', 'f16'])

    assert_equal 'f16', @commands.first.last
  end

  def test_no_chat_template_flag_leaves_out_the_chat_template
    editor = File.join(@llama_cpp, RPEFT::GGUF::METADATA_EDITOR)
    FileUtils.mkdir_p(File.dirname(editor))
    File.write(editor, "# editor\n")
    RPEFT::GGUF.runner = lambda do |argv, _env|
      @commands << argv
      File.binwrite(argv.include?('--outfile') ? argv[argv.index('--outfile') + 1] : argv.last, 'GGUF')
      ['', true]
    end

    status, = run_cli(['export', 'gguf', '--model', @model_dir, '--output', @output, '--llama-cpp', @llama_cpp, '--no-chat-template'])

    assert_equal 0, status
    assert_includes @commands.last, '--remove-metadata'
  end

  def test_accepts_the_quantize_type_in_any_case
    status, = run_cli(['export', 'gguf', '--model', @model_dir, '--output', @output, '--llama-cpp', @llama_cpp, '--quantize', 'F16'])

    assert_equal 0, status
    assert_equal 'f16', @commands.first.last
  end

  def test_rejects_an_unknown_quantize_type
    status, _out, err = run_cli(['export', 'gguf', '--model', @model_dir, '--output', @output, '--quantize', 'q2_k'])

    assert_equal 2, status
    assert_includes err, '--quantize'
    assert_empty @commands
  end

  def test_requires_model_and_output
    status, _out, err = run_cli(%w(export gguf))

    assert_equal 2, status
    assert_includes err, '--model'
    assert_includes err, '--output'
  end

  def test_reports_a_missing_llama_cpp_with_exit_status_one
    status, _out, err = run_cli(
      ['export', 'gguf', '--model', @model_dir, '--output', @output, '--llama-cpp', File.join(@workdir, 'none')]
    )

    assert_equal 1, status
    assert_includes err, 'convert_hf_to_gguf.py not found'
  end

  def test_lists_the_export_formats
    status, out, = run_cli(%w(export))

    assert_equal 0, status
    assert_includes out, 'gguf'
  end

  def test_rejects_an_unknown_export_format
    status, _out, err = run_cli(%w(export onnx))

    assert_equal 2, status
    assert_includes err, 'unknown export format'
  end
end
