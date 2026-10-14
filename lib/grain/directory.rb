# frozen_string_literal: true

require 'fileutils'

module Grain
  # @api private
  module Directory
    module_function

    # Moves source to dest. An existing destination is replaced only when it contains one of the marker files,
    # i.e. when it looks like a previously saved artifact of the same kind.
    # Anything else is refused so that a mistyped path never deletes unrelated data.
    def move(source, dest, replace_markers: [])
      dest = File.expand_path(dest)
      if File.exist?(dest)
        unless replaceable?(dest, replace_markers)
          raise Error, "destination already exists: #{dest} (Grain only replaces " \
                       'directories it previously saved; remove it or pick another path)'
        end

        FileUtils.rm_rf(dest)
      end

      FileUtils.mkdir_p(File.dirname(dest))
      FileUtils.mv(source, dest)

      dest
    end

    # Makes dir ready to be written into, under the same rule as move: it must be missing, empty,
    # or an artifact Grain saved there before.
    def prepare(dir, markers:)
      dir = File.expand_path(dir)
      if File.exist?(dir) && !(File.directory?(dir) && (Dir.empty?(dir) || replaceable?(dir, markers)))
        raise Error, "output directory is not empty: #{dir} (Grain only writes into empty directories " \
                     'or ones it previously wrote; remove it or pick another path)'
      end

      FileUtils.mkdir_p(dir)
      dir
    end

    def replaceable?(dest, markers)
      markers.any? { |marker| File.exist?(File.join(dest, marker)) }
    end
  end
end
