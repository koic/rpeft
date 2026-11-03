# frozen_string_literal: true

require 'fileutils'

module RPEFT
  # @api private
  module Directory
    module_function

    # Moves source to dest. An existing destination is replaced only when it contains one of the marker files,
    # i.e. when it looks like a previously saved artifact of the same kind.
    # Anything else is refused so that a mistyped path never deletes unrelated data.
    def move(source, dest, replace_markers: [])
      # A Model made from a Hugging Face Hub id has no local directory to move.
      raise Error, "nothing to save: #{source} is not a local directory" unless File.directory?(source)

      dest = File.expand_path(dest)
      return dest if canonical(source) == canonical(dest)
      if inside?(source, dest) || inside?(dest, source)
        raise ConfigurationError, "cannot save #{File.expand_path(source)} to #{dest}: one directory contains the other"
      end

      if File.exist?(dest)
        unless replaceable?(dest, replace_markers)
          raise ConfigurationError, "destination already exists: #{dest} (RPEFT only replaces " \
                                    'directories it previously saved; remove it or pick another path)'
        end

        FileUtils.rm_rf(dest)
      end

      FileUtils.mkdir_p(File.dirname(dest))
      FileUtils.mv(source, dest)

      dest
    end

    # Makes dir ready to be written into, under the same rule as move: it must be missing, empty,
    # or an artifact RPEFT saved there before.
    def prepare(dir, markers:)
      dir = File.expand_path(dir)
      if File.exist?(dir) && !(File.directory?(dir) && (Dir.empty?(dir) || replaceable?(dir, markers)))
        raise ConfigurationError, "output directory is not empty: #{dir} (RPEFT only writes into empty directories " \
                                  'or ones it previously wrote; remove it or pick another path)'
      end

      FileUtils.mkdir_p(dir)
      dir
    end

    def replaceable?(dest, markers)
      markers.any? { |marker| File.exist?(File.join(dest, marker)) }
    end

    # Resolves symlinks in the part of the path that exists, so that two spellings of the same location
    # (such as /var and /private/var on macOS) compare equal even when the last components do not exist yet.
    def canonical(path)
      path = File.expand_path(path)
      return File.realpath(path) if File.exist?(path)

      parent = File.dirname(path)
      parent == path ? path : File.join(canonical(parent), File.basename(path))
    end

    def inside?(path, dir)
      canonical(path).start_with?("#{canonical(dir).chomp(File::SEPARATOR)}#{File::SEPARATOR}")
    end
  end
end
